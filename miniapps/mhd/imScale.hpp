#include "mfem.hpp"
#include "PetscPreconditioner.hpp"
#include "InitialConditions.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace mfem;

double variVisc(const Vector &x)
{
    if ((1.-abs(x(1)))<1e-2)
        return 1e-2;
    else
        return 0.;
}

//------------this is for explicit solver only------------
int ex_supg=2;  //1: test supg with v term only (it assumes viscosity==resistivity now)
                //2: test hyperdiffusion along B only 
                //3: test a general hyperdiffusion 
                
//------------this is for implicit solver only------------
bool usesupg=false;  //add supg in both psi and omega
int im_supg=1;
bool usefd=false;   //add field-line diffusion for psi in implicit solvers

int iUpdateJ=1;         //control how J is computed (whether or not Dirichelt boundary condition
                        //is forced at physical boundary, preconditioner prefers enforcing boundary)
                        //2 - using a lumped mass matrix for iupdateJ=1
bool lumpedMass = false;    //use lumped mass matrix in M_solver2
int BgradJ=1;   // B.gradJ operator: 1 (B.grad J, phi)
                //                    2 (-J, B.grad phi)
                //                    3 (-BJ, grad phi)
                // 2 and 3 should be equivalent 
                
int itau_=2;    //how to evaluate supg coefficient

bool pa=false;  //partial assembly in some operators (need to find a way to accelerate supg operators)
                

//------------this is for preconditioner------------
int iSc=0;     //the parameter to control precondtioner
int useFull=1; // control version of preconditioner 
               // 0: a simple block preconditioner
               // 1: physics-based preconditioner
               // 2: physics-based but supg more complicated version
int i_supgpre=3;    //3 - full diagonal supg terms on psi and phi
                    //0 - only (v.grad) in the preconditioner on psi and phi

int bctype = 1; //1 - Dirichlet 
                //2 - weak Dirichlet
double weakPenalty;

double factormin=8.; 
bool debug=false;

int icycle=0;

extern int icase;
extern ParMesh *pmesh;
extern int order;


// reduced system 
class ReducedSystemOperator : public Operator
{
private:
   ParFiniteElementSpace &fespace;
   ParBilinearForm *M, *Mlumped, *K, *KB, *DRe, *DSl, *bdrForm; 
   HypreParMatrix &Mmat, &Mfullmat, &Kmat, *DRematpr, *DSlmatpr, &KBMat;
   //own by this:
   HypreParMatrix *Mdtpr, *ARe, *ASl, *MinvKB;
   mutable HypreParMatrix *ScFull, *AReFull, *NbFull, *PwMat, Mmatlp, *NbMat;
   mutable HypreParMatrix *tmp1, *tmp2;
   bool initialMdt;
   HypreParVector *E0Vec;
   mutable ParLinearForm *StabE0; //source terms
   FunctionCoefficient *E0rhs;
   ParGridFunction *j0;
   Array<int> block_trueOffsets;

   CGSolver *M_solver, *M_solver2, *M_solver3;

   ParaViewDataCollection *pd;

   int myid;
   double dt, dtOld, viscosity, resistivity;
   const Vector *phi, *psi, *w;
   const Array<int> &ess_tdof_list;
   const Array<int> &ess_bdr;

   mutable ParGridFunction phiGf, psiGf, wGf, gftmp,gftmp2,gftmp3, phiOld;
   mutable MyCoefficient *vOld;
   mutable ParBilinearForm *Nv, *Nb, *Pw;
   mutable ParBilinearForm *StabMass, *StabNb, *StabNv; //for stablize B term
   mutable ParLinearForm *PB_VPsi, *PB_VOmega, *PB_BJ;
   mutable BlockOperator *Jacobian;
   mutable Vector z, zdiff, z2, z3, J, zFull;

public:
   ReducedSystemOperator(ParFiniteElementSpace &f,
                         ParBilinearForm *M_, HypreParMatrix &Mmat_, HypreParMatrix &Mfullmat_,
                         ParBilinearForm *K_, HypreParMatrix &Kmat_,
                         ParBilinearForm *KB_, HypreParMatrix &KBMat_,
                         ParBilinearForm *DRe_, ParBilinearForm *DSl_,
                         ParBilinearForm *Mlumped_, HypreParMatrix &MlumpedMat,
                         CGSolver *M_solver_, CGSolver *M_solver2_, CGSolver *M_solver3_,
                         const double visc, const double resi,
                         const Array<int> &ess_tdof_list_,const Array<int> &ess_bdr_);

   //this add the useFull option
   ReducedSystemOperator(ParFiniteElementSpace &f,
                         ParBilinearForm *M_, HypreParMatrix &Mmat_, HypreParMatrix &Mfullmat_,
                         ParBilinearForm *K_, HypreParMatrix &Kmat_,
                         ParBilinearForm *KB_, HypreParMatrix &KBMat_,
                         ParBilinearForm *DRe_, HypreParMatrix *DRemat_,
                         ParBilinearForm *DSl_, HypreParMatrix *DSlmat_,
                         ParBilinearForm *Mlumped_, HypreParMatrix &MlumpedMat,
                         CGSolver *M_solver_, CGSolver *M_solver2_, CGSolver *M_solver3_,
                         const double visc, const double resi,
                         const Array<int> &ess_tdof_list_, const Array<int> &ess_bdr_);

   // Set current values - needed to compute action and Jacobian.
   void SetParameters(double dt_, const Vector *phi_, const Vector *psi_, const Vector *w_)
   {   
       dtOld=dt; dt=dt_; phi=phi_; psi=psi_; w=w_;

       if (vOld==NULL && false)
       {
         Vector &k = const_cast<Vector &>(*phi);
         phiOld.MakeTRef(&fespace, k, 0);
         phiOld.SetFromTrueVector();
         delete vOld;
         vOld = new MyCoefficient(&phiGf, 2);
         if (myid==0) cout <<"------update vOld------"<<endl;
       }

       if (dtOld!=dt && initialMdt)
       {
           if (myid==0) cout <<"------update Mdt------"<<endl;
           double rate=dtOld/dt;
           *Mdtpr*=rate;

           if (useFull > 0)
           {
               delete ARe;
               delete ASl;
               if (DRematpr!=NULL)
                  ARe = ParAdd(Mdtpr, DRematpr);
               else
                  ARe = new HypreParMatrix(*Mdtpr);
                  
               if (DSlmatpr!=NULL)
                  ASl = ParAdd(Mdtpr, DSlmatpr);
               else
                  ASl = new HypreParMatrix(*Mdtpr);
           } 
       }
       if(initialMdt==false)
       {
           if (myid==0) cout <<"------initial Mdt-------"<<endl;
           *Mdtpr*=(1./dt); 
           initialMdt=true;

           if (useFull > 0)
           {
              if (DRematpr!=NULL)
                 ARe = ParAdd(Mdtpr, DRematpr);
              else
                 ARe = new HypreParMatrix(*Mdtpr);
                 
              if (DSlmatpr!=NULL)
                 ASl = ParAdd(Mdtpr, DSlmatpr);
              else
                 ASl = new HypreParMatrix(*Mdtpr);
           }
       }
   }

   void setCurrent(ParGridFunction *gf)
   { j0=gf;}

   //store E0 (rhs) 
   void setE0(HypreParVector *E0Vec_, FunctionCoefficient *E0rhs_)
   { E0Vec=E0Vec_; E0rhs=E0rhs_;}

   /// Define F(k) 
   virtual void Mult(const Vector &k, Vector &y) const;

   /// Define J 
   virtual Operator &GetGradient(const Vector &k) const;

   virtual ~ReducedSystemOperator();

};

//my bchandler (Dirichlet bounary for all the components)
class myBCHandler : public PetscBCHandler
{
private:
    int component, componentSize;
    Vector vx;

public:
    myBCHandler(Array<int>& ess_tdof_list, enum PetscBCHandler::Type _type, 
                int _component, int _componentSize)
   : PetscBCHandler(_type), 
     component(_component), componentSize(_componentSize)
    {
       SetTDofs(ess_tdof_list);
    }

    void SetProblemSize(int component_, int componentSize_)
    {component=component_; componentSize=componentSize_;}

    //overwrite SetTDofs
    void SetTDofs(Array<int>& list)
    {
       int iSize=list.Size();
       //cout <<"======vector size is "<<component<<" "<<iSize<<endl;
       //cout <<"======component size is "<<componentSize<<endl;
       //list.Print();
       //XXX do not apply boundary condition on phi for now

       if (true){
          ess_tdof_list.SetSize(component*iSize);
          for (PetscInt j = 0; j < component; j++)
            for (PetscInt i = 0; i < iSize; i++)
            {
               ess_tdof_list[i+j*iSize] = j*componentSize+list[i];
            }
       }
       else{ 
          ess_tdof_list.SetSize((component-1)*iSize);
          for (PetscInt j = 1; j < component; j++)
            for (PetscInt i = 0; i < iSize; i++)
            {
               ess_tdof_list[i+(j-1)*iSize] = j*componentSize+list[i];
            }
       }
       setup = false;
    }
    void SetBoundary(const Vector &_vx)
    {   
        if (setup) return; 
        vx=_vx;
    }

    void Eval(double t, Vector &g)
    { 
        MFEM_ASSERT(vx.Size()==g.Size(), "size not matched!"); 
        g=0.;
        for (PetscInt i = 0; i < ess_tdof_list.Size(); ++i)
        {
           g[ess_tdof_list[i]] = vx[ess_tdof_list[i]];
        }
    }

    ~myBCHandler() {};
};

// Auxiliary class to provide preconditioners for matrix-free methods 
class FullPreconditionerFactory : public PetscPreconditionerFactory
{
private:
   const ReducedSystemOperator& op;

public:
   FullPreconditionerFactory(const ReducedSystemOperator& op_,
                         const string& name_): PetscPreconditionerFactory(name_), op(op_) {};
   virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle &oh)
   { 
       if(useFull==1)
          return new FullBlockSolver(oh);
       else
          return new SupgBlockSolver(oh);
   }

   virtual ~FullPreconditionerFactory() {};
};

// Auxiliary class to provide preconditioners for matrix-free methods 
class PreconditionerFactory : public PetscPreconditionerFactory
{
private:
   const ReducedSystemOperator& op;

public:
   PreconditionerFactory(const ReducedSystemOperator& op_,
                         const string& name_): PetscPreconditionerFactory(name_), op(op_) {};
   virtual mfem::Solver* NewPreconditioner(const mfem::OperatorHandle &oh)
   { return new MyBlockSolver(oh);}

   virtual ~PreconditionerFactory() {};
};

/** After spatial discretization, the resistive MHD model can be written as a
 *  system of ODEs:
 *     dPsi/dt = M^{-1}*F1,
 *     dw  /dt = M^{-1}*F2,
 *  coupled with two linear systems
 *     j   = -Mfull^{-1}*(K-B)*Psi 
 *     Phi = -K^{-1}*M*w
 *  so far there seems no need to do a BlockNonlinearForm
 *
 *  Class ResistiveMHDOperator represents the right-hand side of the above
 *  system of ODEs. */
class ResistiveMHDOperator : public TimeDependentOperator
{
protected:
   ParFiniteElementSpace &fespace;
   Array<int> ess_tdof_list;

   ParBilinearForm *M, *Mfull, *Mlumped, *K, *KB, *KBform, DSl, DRe; //mass, stiffness, diffusion with SL and Re
   ParBilinearForm *Nv, *Nb;
   mutable ParBilinearForm *StabMass, *StabNb, *StabNv; 
   ParLinearForm *E0, *StabE0; //source terms
   mutable ParLinearForm zLF; //LinearForm holder for updating J
   HypreParMatrix Kmat, Mmat, *Mfullmat, MlumpedMat, DSlmat, DRemat, *KBMat;
   HypreParVector *E0Vec;
   FunctionCoefficient *E0rhs;
   double viscosity, resistivity;
   bool useAMG, use_petsc, use_factory, convergedSolver, tRHS;
   ConstantCoefficient visc_coeff, resi_coeff;
   FunctionCoefficient visc_vari, resi_vari;

   //for implicit stepping
   ReducedSystemOperator *reduced_oper;
   PetscNonlinearSolver *pnewton_solver;
   myBCHandler *bchandler;
   PetscPreconditionerFactory *J_factory;


   int myid;
   CGSolver M_solver; // Krylov solver for inverting the mass matrix M
   HypreSmoother *M_prec;  // Preconditioner for the mass matrix M

   CGSolver M_solver2; // Krylov solver for inverting the mass matrix M
   HypreSmoother *M_prec2;  // Preconditioner for the mass matrix M

   CGSolver M_solver3; // Krylov solver for inverting the mass matrix M (lumped)
   HypreSmoother *M_prec3;  // Preconditioner for the mass matrix M

   CGSolver K_solver; // Krylov solver for inverting the stiffness matrix K
   HypreSmoother *K_prec;  // Preconditioner for the stiffness matrix K

   HypreSolver *K_amg; //BoomerAMG for stiffness matrix
   HyprePCG *K_pcg;

   Vector zero; // empty vector is interpreted as zero r.h.s. by NewtonSolver
   mutable Vector z, J, z2, z3, zFull; // auxiliary vector 
   mutable ParGridFunction j, gftmp, gftmp2, gftmp3;  //auxiliary variable (to store the boundary condition)
   ParBilinearForm *DRetmp, *DSltmp;    //hold the matrices for DRe and DSl

public:
   ResistiveMHDOperator(ParFiniteElementSpace &f, Array<int> &ess_bdr, 
                       double visc, double resi, bool use_petsc_, bool use_factory_); 

   // Compute the right-hand side of the ODE system.
   virtual void Mult(const Vector &vx, Vector &dvx_dt) const;

   //Solve the Backward-Euler equation: k = f(x + dt*k, t), for the unknown k.
   //here vector are block vectors
   virtual void ImplicitSolve(const double dt, const Vector &vx, Vector &k);

   //Update problem in AMR case
   void UpdateProblem(Array<int> &ess_bdr, bool PartialUpdate=false);

   //link gftmp with psi; this is an old way and not needed any more
   void BindingGF(Vector &vx)
   {int sc = height/3; gftmp.MakeTRef(&fespace, vx, sc);}

   void computeV(ParGridFunction *phi, ParGridFunction *v1, ParGridFunction *v2);

   bool getConverged(){ return convergedSolver;}
   void resetConverged(){ convergedSolver=true;}


   //update grid functions (grid functions have to be updated immediately)
   void UpdateGridFunction()
   {
      j.Update(); 
      gftmp.Update();
      //DSl and DRe contains ParGridFunctions that need to be updated
      DSl.Update();    
      DSl.Assemble();
      DRe.Update();    
      DRe.Assemble();
   }

   void outputgf()
   {
      ostringstream gf_name;
      gf_name << "dw_dt." << setfill('0') << setw(6) << myid;
      ofstream osol6(gf_name.str().c_str());
      osol6.precision(8);
      gftmp.Save(osol6);
   }

   //set rhs E0 
   void SetRHSEfield( double(* f)( const Vector&) );
   void SetRHSEfield( double(* f)( const Vector&, double) );
   void SetInitialJ(FunctionCoefficient initJ);

   void UpdateJ(Vector &k, ParGridFunction *jout);

   //functions for explicit solver
   void UpdatePhi(Vector &vx);
   void assembleNv(ParGridFunction *gf);
   void assembleNb(ParGridFunction *gf);
   void assembleVoper(double dt, ParGridFunction *phi, ParGridFunction *psi);
   void assembleBoper(double dt, ParGridFunction *phi, ParGridFunction *psi);

   void DestroyHypre();
   virtual ~ResistiveMHDOperator();
};


ResistiveMHDOperator::ResistiveMHDOperator(ParFiniteElementSpace &f, 
                                         Array<int> &ess_bdr, double visc, double resi, 
                                         bool use_petsc_ = false, bool use_factory_=false)
   : TimeDependentOperator(3*f.TrueVSize(), 0.0), fespace(f),
     M(NULL), Mfull(NULL), Mlumped(NULL), K(NULL), KB(NULL), KBform(NULL), DSl(&fespace), DRe(&fespace),
     Nv(NULL), Nb(NULL), StabMass(NULL), StabNb(NULL), StabNv(NULL),  
     E0(NULL), StabE0(NULL), zLF(&fespace), Mfullmat(NULL), E0Vec(NULL), E0rhs(NULL),
     viscosity(visc),  resistivity(resi), useAMG(false), tRHS(false), use_petsc(use_petsc_), use_factory(use_factory_),
     convergedSolver(true),
     visc_coeff(visc),  resi_coeff(resi),  visc_vari(resiVari),  resi_vari(resiVari),  
     reduced_oper(NULL), pnewton_solver(NULL), bchandler(NULL), J_factory(NULL),
     M_solver(f.GetComm()), M_prec(NULL), M_solver2(f.GetComm()), M_prec2(NULL), M_solver3(f.GetComm()), M_prec3(NULL),
     K_solver(f.GetComm()),  K_prec(NULL),
     K_amg(NULL), K_pcg(NULL), z(height/3), 
     J(height/3), z2(height/3), z3(height/3), zFull(f.GetVSize()),
     j(&fespace), gftmp(&fespace), gftmp2(&fespace), gftmp3(&fespace),
     DRetmp(NULL), DSltmp(NULL)
{
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   //mass matrix
   M = new ParBilinearForm(&fespace);
   M->AddDomainIntegrator(new MassIntegrator);
   M->Assemble();
   M->FormSystemMatrix(ess_tdof_list, Mmat);

   //full mass matrix 
   Mfull = new ParBilinearForm(&fespace);
   MassIntegrator *mass = new MassIntegrator;
   if (lumpedMass) //use a lumped mass integrator to compute J
   {
     if (myid==0) cout <<"------lumped mass matrix in M_solver2!------"<<endl;
     Mfull->AddDomainIntegrator(new LumpedIntegrator(mass));
     Mfull->Assemble();
     Mfull->Finalize();
     Mfullmat=Mfull->ParallelAssemble();
   }
   else 
   {
     Mfull->AddDomainIntegrator(mass);
     Mfull->Assemble();
     Mfull->Finalize();
     Mfullmat=Mfull->ParallelAssemble();
   }

   MassIntegrator *mass2 = new MassIntegrator;
   Mlumped = new ParBilinearForm(&fespace);
   Mlumped->AddDomainIntegrator(new LumpedIntegrator(mass2));
   Mlumped->Assemble();
   Mlumped->FormSystemMatrix(ess_tdof_list, MlumpedMat);

   M_solver.iterative_mode = false;
   M_solver.SetRelTol(1e-7);
   M_solver.SetAbsTol(0.0);
   M_solver.SetMaxIter(2000);
   M_solver.SetPrintLevel(0);
   M_prec = new HypreSmoother;
   M_prec->SetType(HypreSmoother::Jacobi);
   M_solver.SetPreconditioner(*M_prec);
   M_solver.SetOperator(Mmat);

   M_solver2.iterative_mode = false;
   M_solver2.SetRelTol(1e-7);
   M_solver2.SetAbsTol(0.0);
   M_solver2.SetMaxIter(2000);
   M_solver2.SetPrintLevel(0);
   M_prec2 = new HypreSmoother;
   M_prec2->SetType(HypreSmoother::Jacobi);
   M_solver2.SetPreconditioner(*M_prec2);
   M_solver2.SetOperator(*Mfullmat);

   M_solver3.iterative_mode = false;
   M_solver3.SetRelTol(1e-7);
   M_solver3.SetAbsTol(0.0);
   M_solver3.SetMaxIter(2000);
   M_solver3.SetPrintLevel(0);
   M_prec3 = new HypreSmoother;
   M_prec3->SetType(HypreSmoother::Jacobi);
   M_solver3.SetPreconditioner(*M_prec3);
   M_solver3.SetOperator(MlumpedMat);

   //stiffness matrix
   K = new ParBilinearForm(&fespace);
   K->AddDomainIntegrator(new DiffusionIntegrator);
   K->Assemble();
   K->FormSystemMatrix(ess_tdof_list, Kmat);

   useAMG=true;
   if (useAMG)
   {
      K_amg = new HypreBoomerAMG(Kmat);
      K_pcg = new HyprePCG(Kmat);
      K_pcg->iterative_mode = false;
      K_pcg->SetTol(1e-7);
      K_pcg->SetMaxIter(200);
      K_pcg->SetPrintLevel(0);
      K_pcg->SetPreconditioner(*K_amg);
   }
   else
   {
      K_solver.iterative_mode = false;
      K_solver.SetRelTol(1e-7);
      K_solver.SetAbsTol(0.0);
      K_solver.SetMaxIter(2000);
      K_solver.SetPrintLevel(3);
      delete K_prec;
      K_prec = new HypreSmoother;
      K_prec->SetType(HypreSmoother::Chebyshev);
      K_solver.SetPreconditioner(*K_prec);
      K_solver.SetOperator(Kmat);
   }

   KB = new ParBilinearForm(&fespace);
   KB->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   KB->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix
   KB->Assemble();
   KB->Finalize();
   KBMat=KB->ParallelAssemble();

   KBform = new ParBilinearForm(&fespace);
   KBform->AddDomainIntegrator(new DiffusionIntegrator);      //  K matrix
   KBform->AddBdrFaceIntegrator(new BoundaryGradIntegrator);  // -B matrix
   KBform->Assemble();
 

   Coefficient *visc_ptr, *resi_ptr;
   if (icase==5)
   {
       visc_ptr = &visc_vari;
       resi_ptr = &resi_vari;
   }
   else
   {
       visc_ptr = &visc_coeff;
       resi_ptr = &resi_coeff;
   }
   DRe.AddDomainIntegrator(new DiffusionIntegrator(*visc_ptr));    
   DRe.Assemble();

   DSl.AddDomainIntegrator(new DiffusionIntegrator(*resi_ptr));    
   DSl.Assemble();

   if (use_petsc)
   {
      ParBilinearForm *DRepr=NULL, *DSlpr=NULL;
      HypreParMatrix *DRematpr=NULL, *DSlmatpr=NULL;
      if (viscosity != 0.0)
      {   
          //assemble diffusion matrices (cannot delete DRetmp if ParAdd is used later)
          DRetmp = new ParBilinearForm(&fespace);
          DRetmp->AddDomainIntegrator(new DiffusionIntegrator(*visc_ptr));    
          DRetmp->Assemble();
          DRetmp->FormSystemMatrix(ess_tdof_list, DRemat);

          DRematpr = &DRemat;
          DRepr = &DRe;
      }

      if (resistivity != 0.0)
      {
          DSltmp = new ParBilinearForm(&fespace);
          DSltmp->AddDomainIntegrator(new DiffusionIntegrator(*resi_ptr));    
          DSltmp->Assemble();
          DSltmp->FormSystemMatrix(ess_tdof_list, DSlmat);

          DSlmatpr = &DSlmat;
          DSlpr = &DSl;
      }

      reduced_oper  = new ReducedSystemOperator(fespace, M, Mmat, *Mfullmat, K, Kmat, 
                         KBform, *KBMat, DRepr, DRematpr, DSlpr, DSlmatpr, Mlumped, MlumpedMat,
                         &M_solver, &M_solver2, &M_solver3,
                         viscosity, resistivity, ess_tdof_list, ess_bdr);


      const double rel_tol=1e-4;
      pnewton_solver = new PetscNonlinearSolver(fespace.GetComm(),*reduced_oper);
      if (use_factory)
      {
         SNES snes=SNES(*pnewton_solver);

         /*
         KSP ksp; 
		 SNESGetKSP(snes,&ksp);
		 KSPSetType(ksp,KSPFGMRES);
         SNESKSPSetUseEW(snes,PETSC_TRUE);
         SNESKSPSetParametersEW(snes,2,1e-4,0.1,0.9,1.5,1.5,0.1);
         */

         if (useFull>0)
            J_factory = new FullPreconditionerFactory(*reduced_oper, "JFNK Full preconditioner");
         else
            J_factory = new PreconditionerFactory(*reduced_oper, "JFNK preconditioner");
         pnewton_solver->SetPreconditionerFactory(J_factory);
      }
      pnewton_solver->SetPrintLevel(0); // print Newton iterations
      pnewton_solver->SetRelTol(rel_tol);
      pnewton_solver->SetAbsTol(0.0);
      pnewton_solver->SetMaxIter(20);
      pnewton_solver->iterative_mode=true;

      if (bctype==1)
      {
         //3 components in block vector; each has the size of height/3
         bchandler = new myBCHandler(ess_tdof_list, PetscBCHandler::CONSTANT, 3, height/3);
         pnewton_solver->SetBCHandler(bchandler);
      }
   }

   
}

void ResistiveMHDOperator::UpdateProblem(Array<int> &ess_bdr, bool PartialUpdate)
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}

//a time dependent rhs
void ResistiveMHDOperator::SetRHSEfield( double(* f)( const Vector&,  double t) ) 
{
   tRHS=true;
   delete E0;
   delete E0rhs;
   E0rhs = new FunctionCoefficient(f);
   E0 = new ParLinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(*E0rhs));
}

void ResistiveMHDOperator::SetRHSEfield( double(* f)( const Vector&) ) 
{
   delete E0;
   delete E0rhs;
   E0rhs = new FunctionCoefficient(f);
   E0 = new ParLinearForm(&fespace);
   E0->AddDomainIntegrator(new DomainLFIntegrator(*E0rhs));
   E0->Assemble();
   E0Vec=E0->ParallelAssemble();

   //add E0 to reduced_oper
   if (reduced_oper!=NULL)
      reduced_oper->setE0(E0Vec, E0rhs);
}

void ResistiveMHDOperator::SetInitialJ(FunctionCoefficient initJ) 
{
    j.ProjectCoefficient(initJ);
    j.SetTrueVector();
    j.SetFromTrueVector();

    //add current to reduced_oper 
    if (reduced_oper!=NULL)
        reduced_oper->setCurrent(&j);
}
   
void ResistiveMHDOperator::computeV(ParGridFunction *phi, ParGridFunction *v1, ParGridFunction *v2)
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}

void ResistiveMHDOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}

void ResistiveMHDOperator::ImplicitSolve(const double dt,
                                         const Vector &vx, Vector &k)
{
   double time=GetTime();
   if(tRHS)
   {   
      if (myid==0) cout<<"++++++ update source at time = "<<time<<" ++++++\n";
      E0rhs->SetTime(time);
      E0->Assemble();
      E0Vec=E0->ParallelAssemble();

      //add E0 to reduced_oper
      if (reduced_oper!=NULL)
      {  reduced_oper->setE0(E0Vec, E0rhs); }
   }
   
   int sc = height/3;
   Vector phi(vx.GetData() +   0, sc);
   Vector psi(vx.GetData() +  sc, sc);
   Vector   w(vx.GetData() +2*sc, sc);

   reduced_oper->SetParameters(dt, &phi, &psi, &w);
   
   k = vx; //Provide the initial guess as vx and use iterative_mode

   if (bctype==1)
   {
      bchandler->SetBoundary(vx);   //setup the essential boundary (in the first solve)
   }

   //we skip the current solve if convergedSolver is not true (happens in later stage of RK)
   if (!convergedSolver)
   {
       if (myid==0) cout<<"======WARNING: Previous ImplicitSolve did not converge. Skip current ImplicitSolve until it gets reset!======\n";
       return;
   }

   if (bctype!=1 && zero.Size()==0)
   {
       if (myid==0) cout<<"====== Initialize the rhs in ImplicitSolve ======\n";

       zero.SetSize(height);
       zero=0.;

       Vector y2(zero.GetData() +  sc, sc);
       Vector y3(zero.GetData() +2*sc, sc);

       ParLinearForm rhs2(&fespace);
       FunctionCoefficient psiBC(InitialPsi3);
 
       rhs2.AddBoundaryIntegrator(new BoundaryLFIntegrator(psiBC));
       rhs2.Assemble();
       rhs2.ParallelAssemble(y2);

       zero*=weakPenalty;
   }

   pnewton_solver->Mult(zero, k);  //here k is solved as vx^{n+1}
    
   if (pnewton_solver->GetConverged())
   {
      convergedSolver=true;
   }
   else
   {
      if (myid==0) cout<<"======WARNING: Newton solver did not converge. reduce timestep!======\n";
      convergedSolver=false;
   }
   //modify k so that it fits into the backward euler framework
   k-=vx;
   k/=dt;
   
}

void ResistiveMHDOperator::assembleVoper(double dt, ParGridFunction *phi, ParGridFunction *psi) 
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}

void ResistiveMHDOperator::assembleBoper(double dt, ParGridFunction *phi, ParGridFunction *psi) 
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}


void ResistiveMHDOperator::assembleNv(ParGridFunction *gf) 
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}

void ResistiveMHDOperator::assembleNb(ParGridFunction *gf) 
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}

void ResistiveMHDOperator::UpdatePhi(Vector &vx)
{
      MFEM_ABORT("Error: wrong option in implicit scaling test!"); 
}

void ResistiveMHDOperator::UpdateJ(Vector &k, ParGridFunction *jout)
{
   //the current is J=-M^{-1}*K*Psi
   int sc = height/3;
   Vector psi(k.GetData() + sc, sc);

   if (iUpdateJ==0)
   {
      KBMat->Mult(psi, z);
      z.Neg();
      M_solver2.Mult(z, J);
   }
   else if (iUpdateJ==1)
   {
      gftmp.SetFromTrueDofs(psi);
      Vector Z;
      HypreParMatrix A;
      KBform->Mult(gftmp, zFull);
      zFull.Neg(); // z = -z
      M->FormLinearSystem(ess_tdof_list, j, zFull, A, J, Z); //apply Dirichelt boundary 
      M_solver.Mult(Z, J); 
   }
   else if (iUpdateJ==2)
   {
      gftmp.SetFromTrueDofs(psi);
      Vector Z;
      HypreParMatrix A;
      KBform->Mult(gftmp, zFull);
      zFull.Neg(); // z = -z
      Mlumped->FormLinearSystem(ess_tdof_list, j, zFull, A, J, Z); //apply Dirichelt boundary 
      M_solver3.Mult(Z, J); 
   }
   else
      MFEM_ABORT("Error: wrong option of iUpdateJ"); 
   
   jout->SetFromTrueDofs(J);
}

void ResistiveMHDOperator::DestroyHypre()
{
    //hypre and petsc needs to be deleted earilier
    delete K_amg;
    delete M_prec;
    delete M_prec2;
    delete M_prec3;
    delete K_prec;
    delete reduced_oper;
    delete J_factory;
    delete pnewton_solver;
    delete bchandler;
}

ResistiveMHDOperator::~ResistiveMHDOperator()
{
    //free used memory
    delete M;
    delete Mfull;
    delete Mfullmat;
    delete Mlumped;
    delete K;
    delete KBMat;
    delete E0;
    delete E0Vec;
    delete E0rhs;
    delete KB;
    delete KBform;
    delete Nv;
    delete Nb;
    delete StabNv;
    delete StabNb;
    delete StabMass;
    delete StabE0;
    delete K_pcg;
    delete DRetmp;
    delete DSltmp;
}

ReducedSystemOperator::ReducedSystemOperator(ParFiniteElementSpace &f,
   ParBilinearForm *M_, HypreParMatrix &Mmat_, HypreParMatrix &Mfullmat_,
   ParBilinearForm *K_, HypreParMatrix &Kmat_,
   ParBilinearForm *KB_, HypreParMatrix &KBMat_,
   ParBilinearForm *DRe_, ParBilinearForm *DSl_,
   ParBilinearForm *Mlumped_, HypreParMatrix &MlumpedMat,
   CGSolver *M_solver_, CGSolver *M_solver2_,CGSolver *M_solver3_,
   const double visc, const double resi,
   const Array<int> &ess_tdof_list_, const Array<int> &ess_bdr_)
   : Operator(3*f.TrueVSize()), fespace(f), 
     M(M_), K(K_), KB(KB_), DRe(DRe_), DSl(DSl_), Mmat(Mmat_), Mfullmat(Mfullmat_), Kmat(Kmat_), KBMat(KBMat_),
     initialMdt(false), E0Vec(NULL), StabE0(NULL), E0rhs(NULL), Mlumped(Mlumped_), Mmatlp(MlumpedMat),
     M_solver(M_solver_), M_solver2(M_solver2_),M_solver3(M_solver3_),
     dt(0.0), dtOld(0.0), viscosity(visc), resistivity(resi), 
     phi(NULL), psi(NULL), w(NULL), vOld(NULL),
     ess_tdof_list(ess_tdof_list_),ess_bdr(ess_bdr_), gftmp(&fespace),gftmp2(&fespace), gftmp3(&fespace),
     Nv(NULL), Nb(NULL), Pw(NULL), 
     StabMass(NULL), StabNb(NULL), StabNv(NULL),
     PB_VPsi(NULL), PB_VOmega(NULL), PB_BJ(NULL),
     Jacobian(NULL), z(height/3), zdiff(height/3), z2(height/3), z3(height/3), 
     J(height/3), zFull(f.GetVSize())
{ 
    useFull=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    //the following is not right because Mdtpr shares the same matrix with Mmat_
    //hypre_ParCSRMatrix *csrM = (hypre_ParCSRMatrix*)(Mmat_);
    //Mdtpr = new HypreParMatrix(csrM, true);

    //correct way to deep copy:
    Mdtpr = new HypreParMatrix(Mmat_);

    int sc = height/3;
    block_trueOffsets.SetSize(4);
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = sc;
    block_trueOffsets[2] = 2*sc;
    block_trueOffsets[3] = 3*sc;

    if (debug)
    {
      pd = new ParaViewDataCollection("debug-resi", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("residual1", &gftmp);
      pd->RegisterField("residual2", &gftmp2);
      pd->RegisterField("residual3", &gftmp3);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(icycle);
    }
    else
       pd=NULL;
}

ReducedSystemOperator::ReducedSystemOperator(ParFiniteElementSpace &f,
   ParBilinearForm *M_, HypreParMatrix &Mmat_,HypreParMatrix &Mfullmat_,
   ParBilinearForm *K_, HypreParMatrix &Kmat_,
   ParBilinearForm *KB_,HypreParMatrix &KBMat_, 
   ParBilinearForm *DRe_, HypreParMatrix *DRemat_,
   ParBilinearForm *DSl_, HypreParMatrix *DSlmat_,
   ParBilinearForm *Mlumped_, HypreParMatrix &MlumpedMat,
   CGSolver *M_solver_, CGSolver *M_solver2_,CGSolver *M_solver3_,
   const double visc, const double resi,
   const Array<int> &ess_tdof_list_, const Array<int> &ess_bdr_)
   : Operator(3*f.TrueVSize()), fespace(f), 
     M(M_), K(K_), KB(KB_), DRe(DRe_), DSl(DSl_), Mmat(Mmat_), Mfullmat(Mfullmat_), Kmat(Kmat_), KBMat(KBMat_),
     initialMdt(false),E0Vec(NULL), StabE0(NULL), E0rhs(NULL),
     Mlumped(Mlumped_), Mmatlp(MlumpedMat),
     M_solver(M_solver_), M_solver2(M_solver2_), M_solver3(M_solver3_), 
     dt(0.0), dtOld(0.0), viscosity(visc), resistivity(resi),
     phi(NULL), psi(NULL), w(NULL), vOld(NULL),
     ess_tdof_list(ess_tdof_list_), ess_bdr(ess_bdr_), gftmp(&fespace),gftmp2(&fespace), gftmp3(&fespace),
     Nv(NULL), Nb(NULL), Pw(NULL),  
     StabMass(NULL), StabNb(NULL), StabNv(NULL),
     PB_VPsi(NULL), PB_VOmega(NULL), PB_BJ(NULL),
     Jacobian(NULL), z(height/3), zdiff(height/3), z2(height/3), z3(height/3), 
     J(height/3), zFull(f.GetVSize())
{ 
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    Mdtpr = new HypreParMatrix(Mmat_);
    ARe=NULL; ASl=NULL;

    DRematpr = DRemat_;
    DSlmatpr = DSlmat_;

    AReFull=NULL; ScFull=NULL; NbFull=NULL; PwMat=NULL; NbMat=NULL; MinvKB=NULL;
    tmp1=NULL; tmp2=NULL;

    if (usefd || usesupg)
    {
       MinvKB = new HypreParMatrix(KBMat_);
       HypreParVector *MmatlpD = new HypreParVector(Mmatlp.GetComm(), Mmatlp.GetGlobalNumRows(),
                                     Mmatlp.GetRowStarts());
       Mmatlp.GetDiag(*MmatlpD);
       MinvKB->InvScaleRows(*MmatlpD);
       delete MmatlpD;
    }

    int sc = height/3;
    block_trueOffsets.SetSize(4);
    block_trueOffsets[0] = 0;
    block_trueOffsets[1] = sc;
    block_trueOffsets[2] = 2*sc;
    block_trueOffsets[3] = 3*sc;

   if (debug)
   {
      pd = new ParaViewDataCollection("debug-resi", pmesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("residual1", &gftmp);
      pd->RegisterField("residual2", &gftmp2);
      pd->RegisterField("residual3", &gftmp3);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      pd->SetHighOrderOutput(true);
      pd->SetCycle(icycle);
   }
   else
       pd=NULL;

   bdrForm = new ParBilinearForm(&fespace);
   if (bctype==1){
       FunctionCoefficient visc_coeff(variVisc);
       bdrForm->AddDomainIntegrator(new DiffusionIntegrator(visc_coeff));    
   }
   else{
       ConstantCoefficient penalty_coeff(weakPenalty);
       bdrForm->AddBoundaryIntegrator(new BoundaryMassIntegrator(penalty_coeff));    
   }
   bdrForm->Assemble();

}

/*
 * the full preconditioner is (note the sign of Nb)
 * [  ARe Nb  (Mlp)]
 * [  Pw  Sc  0    ]
 * [  K   0   M    ]
*/
Operator &ReducedSystemOperator::GetGradient(const Vector &k) const
{
   MFEM_ASSERT(initialMdt, "Mdt not initialized correctly!"); 

   if (useFull>0)
   {
       delete Jacobian;
       delete AReFull; 
       delete ScFull; 
       delete NbFull;
       delete PwMat;
       delete tmp1;
       delete tmp2;

       Vector &k_ = const_cast<Vector &>(k);

       int sc = height/3;

       //form Nv matrix
       delete Nv;
       phiGf.MakeTRef(&fespace, k_, 0);
       phiGf.SetFromTrueVector();
       Nv = new ParBilinearForm(&fespace);
       MyCoefficient velocity(&phiGf, 2);   //we update velocity
       Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
       Nv->Assemble(); 
       Nv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
       Nv->Finalize();
       HypreParMatrix *NvMat = Nv->ParallelAssemble();

       //change AReFull to the true ARe operator and ScFull to the true ASl operator
       AReFull = ParAdd(ARe, NvMat);
       HypreParMatrix *ASltmp = ParAdd(ASl, NvMat);    

       //form Nb matrix
       delete Nb;
       psiGf.MakeTRef(&fespace, k_, sc);
       psiGf.SetFromTrueVector();
       Nb = new ParBilinearForm(&fespace);
       MyCoefficient Bfield(&psiGf, 2);   //we update B
       Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
       Nb->Assemble();
       Nb->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
       Nb->Finalize();
       NbFull = Nb->ParallelAssemble();

       //form Pw operator        
       delete Pw;
       wGf.MakeTRef(&fespace, k_, 2*sc);
       wGf.SetFromTrueVector();
       Pw = new ParBilinearForm(&fespace);
       MyCoefficient curlw(&wGf, 2);
       Pw->AddDomainIntegrator(new ConvectionIntegrator(curlw));
       Pw->Assemble();
       Pw->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
       Pw->Finalize();
       PwMat = Pw->ParallelAssemble();

       //here we use B^T D^-1 B = (D^-1 B)^T B
       HypreParMatrix *DinvNb = new HypreParMatrix(*NbFull);
       HypreParVector *ARed = new HypreParVector(AReFull->GetComm(), AReFull->GetGlobalNumRows(),
                                        AReFull->GetRowStarts());
       HypreParMatrix *NbtDinv=NULL, *S=NULL;

       if (useFull==1)
       {
         if (iSc==0)
         {
             if (usesupg && i_supgpre==0 && im_supg!=7)           
             {
                  delete StabNv;
                  StabNv = new ParBilinearForm(&fespace);
                  StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity,itau_));
                  StabNv->Assemble(); 
                  StabNv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                  StabNv->Finalize();
                  HypreParMatrix *MatStabNv=StabNv->ParallelAssemble();

                  HypreParMatrix *tmp=ParAdd(ASltmp, MatStabNv);
                  delete ASltmp;
                  ASltmp=tmp;

                  if ( im_supg>0 && im_supg<4 )
                  {
                      if (resistivity!=viscosity || itau_!=2)
                      {
                          delete StabNv;
                          StabNv = new ParBilinearForm(&fespace);
                          StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
                          StabNv->Assemble(); 
                          StabNv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                          StabNv->Finalize();
                          delete MatStabNv;
                          MatStabNv=StabNv->ParallelAssemble();
                      }

                      tmp=ParAdd(AReFull, MatStabNv);
                      delete AReFull;
                      AReFull=tmp;
                  }
                  delete MatStabNv;
             }
             else if (usesupg && i_supgpre>0 && im_supg!=7)
             {
                  delete StabMass;
                  StabMass = new ParBilinearForm(&fespace);
                  if (dtfactor > factormin && itau_!=2)
                  { 
                     if (myid==0) 
                            cout <<"======WARNING: use factormin in tau formula"<<endl;
                      StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, resistivity, velocity, itau_, factormin)); 
                  }
                  else
                  { StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, resistivity, velocity, itau_)); }
                  StabMass->Assemble(); 
                  StabMass->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                  StabMass->Finalize();
                  HypreParMatrix *MatStabMass=StabMass->ParallelAssemble();

                  delete StabNv;
                  StabNv = new ParBilinearForm(&fespace);
                  if (dtfactor > factormin && itau_!=2)
                  { StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity, itau_, factormin)); }
                  else
                  { StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity, itau_)); }
                  StabNv->Assemble(); 
                  StabNv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                  StabNv->Finalize();
                  HypreParMatrix *MatStabNv=StabNv->ParallelAssemble();

                  S = ParMult(MatStabMass, MinvKB);
                  *S *= resistivity;
                  HypreParMatrix *tmp=Add(1./dt, *MatStabMass, 1., *MatStabNv);
                  HypreParMatrix *MatStabSum=ParAdd(tmp, S);

                  delete S;
                  delete tmp;
                  tmp=ParAdd(ASltmp, MatStabSum);
                  delete ASltmp;
                  ASltmp=tmp;

                  if (i_supgpre>2 && ( im_supg==1 || im_supg==3 ) )
                  {
                      if (resistivity!=viscosity || itau_!=2)
                      {
                          if (myid==0) 
                            cout <<"======WARNING: assemble different supg diagonal operators in psi and phi======"<<endl;

                          delete StabMass;
                          StabMass = new ParBilinearForm(&fespace);
                          StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, viscosity, velocity));
                          StabMass->Assemble(); 
                          StabMass->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                          StabMass->Finalize();
                          delete MatStabMass;
                          MatStabMass=StabMass->ParallelAssemble();

                          delete StabNv;
                          StabNv = new ParBilinearForm(&fespace);
                          StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
                          StabNv->Assemble(); 
                          StabNv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                          StabNv->Finalize();
                          delete MatStabNv;
                          MatStabNv=StabNv->ParallelAssemble();

                          S = ParMult(MatStabMass, MinvKB);
                          *S *= viscosity;
                          tmp=Add(1./dt, *MatStabMass, 1., *MatStabNv);
                          delete MatStabSum;
                          MatStabSum=ParAdd(tmp, S);

                          delete S;
                          delete tmp;
                      }
                      tmp=ParAdd(AReFull, MatStabSum);
                      delete AReFull;
                      AReFull=tmp;
                  }
                  else if (i_supgpre==2 && ( im_supg==1 || im_supg==3 ) )
                  {
                      if (resistivity!=viscosity || itau_!=2)
                      {
                          if (myid==0) 
                            cout <<"======WARNING: assemble different supg diagonal operators for psi and phi======"<<endl;

                          MFEM_ABORT("Error in preconditioner: I will not support this option for now"); 
                      }
                      tmp=ParAdd(AReFull, MatStabNv);
                      delete AReFull;
                      AReFull=tmp;
                  }

                  delete MatStabMass;
                  delete MatStabNv;
                  delete MatStabSum;
             }
             else if (usesupg && im_supg==7)           
             {
                  delete StabNv;
                  StabNv = new ParBilinearForm(&fespace);
                  StabNv->AddDomainIntegrator(new SpecialConvectionIntegrator(dt, resistivity, velocity, itau_));
                  StabNv->Assemble(); 
                  StabNv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                  StabNv->Finalize();
                  HypreParMatrix *MatStabNv=StabNv->ParallelAssemble();

                  HypreParMatrix *tmp=ParAdd(ASltmp, MatStabNv);
                  delete ASltmp;
                  ASltmp=tmp;

                  delete StabNv;
                  StabNv = new ParBilinearForm(&fespace);
                  StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
                  StabNv->Assemble(); 
                  StabNv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                  StabNv->Finalize();
                  delete MatStabNv;
                  MatStabNv=StabNv->ParallelAssemble();

                  tmp=ParAdd(AReFull, MatStabNv);
                  delete AReFull;
                  AReFull=tmp;
                  delete MatStabNv;
             }

             //VERSION0: same as Luis's preconditioner
             AReFull->GetDiag(*ARed);
             DinvNb->InvScaleRows(*ARed);
             NbtDinv=DinvNb->Transpose();
             S = ParMult(NbtDinv, NbFull);
             ScFull = ParAdd(ASltmp, S);

             if (usefd)
             {
                delete StabNb;
                StabNb = new ParBilinearForm(&fespace);
                StabNb->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, Bfield, true));
                StabNb->Assemble(); 
                StabNb->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
                StabNb->Finalize();
                HypreParMatrix *MatStabNb=StabNb->ParallelAssemble();

                tmp2 = ParMult(MatStabNb, MinvKB);
                tmp1 = ParAdd(ScFull, tmp2);
                delete ScFull;
                ScFull=tmp1;
                tmp1=NULL;
                delete MatStabNb;
             }

             /*
             if (false && im_supg==1 && usesupg)
             {
              if (myid==0 && false) cout <<"======WARNING: use preconditioner with terms on ARe======"<<endl;
                  HypreParMatrix *MatStabNv=StabNv->ParallelAssemble();
                  tmp=ParAdd(AReFull, MatStabNv);
                  delete AReFull;
                  AReFull=tmp;
             }
             */
         }
         /*
         else if (iSc==0 && usefd && !usesupg)
         {
             //VERSION3: Luis's preconditioner + hyperdiffusion
             AReFull->GetDiag(*ARed);
             DinvNb->InvScaleRows(*ARed);
             NbtDinv=DinvNb->Transpose();
             S = ParMult(NbtDinv, NbFull);
             HypreParMatrix *ScFull1 = ParAdd(ASltmp, S);

             delete StabNb;
             StabNb = new ParBilinearForm(&fespace);
             StabNb->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, Bfield, true));
             StabNb->Assemble(); 
             StabNb->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
             StabNb->Finalize();
             HypreParMatrix *MatStabNb=StabNb->ParallelAssemble();

             delete S;
             S = ParMult(MatStabNb, MinvKB);
             ScFull = ParAdd(ScFull1, S);
             delete ScFull1;
             delete MatStabNb;
         }
         */
         else if (iSc==1)
         {
             //VERSION1: schur complement without transpose
             //here Sc=ASl-B D^-1 B 
             AReFull->GetDiag(*ARed);
             DinvNb->InvScaleRows(*ARed);
             S = ParMult(NbFull, DinvNb);
             *S *= -1;
             ScFull = ParAdd(ASltmp, S);
         }
         else if (iSc==2) {
             //VERSION2: use (lumped) mass matrix
             if (myid==0) cout <<"======WARNING: use scaled mass matrix in Schur complement. this changes preconditioner in pcshell======"<<endl;
             
             Mdtpr->GetDiag(*ARed);
             DinvNb->InvScaleRows(*ARed);
             NbtDinv=DinvNb->Transpose();
             S = ParMult(NbtDinv, NbFull);
             ScFull = ParAdd(ASltmp, S);
         } 
         else 
             MFEM_ABORT("Error in preconditioner."); 

         Jacobian = new BlockOperator(block_trueOffsets);
         Jacobian->SetBlock(0,0,AReFull);
         Jacobian->SetBlock(0,1,NbFull);
         Jacobian->SetBlock(1,0,PwMat);
         Jacobian->SetBlock(1,1,ScFull);
         Jacobian->SetBlock(2,0,&Kmat);
         Jacobian->SetBlock(2,2,&Mmat);

         if (iSc==2) Jacobian->SetBlock(0,2,Mdtpr);
       }
       else
       {
         if (iSc!=0 || !usesupg || usefd)
            MFEM_ABORT("ERROR in preconditioner: wrong option!"); 

         //more complicated version to handle stabilization terms
         delete StabMass;
         StabMass = new ParBilinearForm(&fespace);
         StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, resistivity, velocity));
         StabMass->Assemble(); 
         StabMass->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
         StabMass->Finalize();
         HypreParMatrix *MatStabMass=StabMass->ParallelAssemble();

         delete StabNv;
         StabNv = new ParBilinearForm(&fespace);
         StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity));
         StabNv->Assemble(); 
         StabNv->EliminateEssentialBC(ess_bdr, Matrix::DIAG_ZERO);
         StabNv->Finalize();
         HypreParMatrix *MatStabNv=StabNv->ParallelAssemble();

         S = ParMult(MatStabMass, MinvKB);
         *S *= viscosity;

         tmp1=Add(1./dt, *MatStabMass, 1., *MatStabNv);
         tmp2=ParAdd(tmp1, S);  //tmp2 is the stablization term

         delete S;
         delete MatStabMass;
         delete MatStabNv;
         delete tmp1;

         tmp1=ParAdd(ASltmp, tmp2);
         delete ASltmp;
         ASltmp=tmp1;
         tmp1=NULL;

         if (resistivity!=viscosity)
         {
             MFEM_ABORT("Error in preconditioner. Need to assemble another MinvKB"); 
         }

         //VERSION0: same as Luis's preconditioner
         AReFull->GetDiag(*ARed);
         DinvNb->InvScaleRows(*ARed);
         NbtDinv=DinvNb->Transpose();
         S = ParMult(NbtDinv, NbFull);
         ScFull = ParAdd(ASltmp, S);

         Jacobian = new BlockOperator(block_trueOffsets);
         Jacobian->SetBlock(0,0,AReFull);
         Jacobian->SetBlock(0,1,NbFull);
         Jacobian->SetBlock(0,2,tmp2);
         Jacobian->SetBlock(1,0,PwMat);
         Jacobian->SetBlock(1,1,ScFull);
         Jacobian->SetBlock(2,0,&Kmat);
         Jacobian->SetBlock(2,2,&Mmat);
       }

       bool outputMatrix=false;
       if (outputMatrix)
       {
           if (myid==0) cout <<"======OUTPUT: matrices in ReducedSystemOperator:GetGradient======"<<endl;

           ofstream myf ("DRe.m");
           DRematpr->PrintMatlab(myf);

           ofstream myfile ("ARe.m");
           ARe->PrintMatlab(myfile);

           ofstream myfile0 ("AReFull.m");
           AReFull->PrintMatlab(myfile0);

           ofstream myfile2 ("NvMat.m");
           NvMat->PrintMatlab(myfile2);

           ofstream myfile4 ("lump.m");
           Mmatlp.PrintMatlab(myfile4);

           ARed->Print("diag.dat");
       }

       delete DinvNb;
       delete ARed;
       delete NbtDinv;
       delete S;
       delete NvMat;
       delete ASltmp;


   }
   else
   {
      if (Jacobian == NULL)    //in the first pass we just set Jacobian once
      {
         Jacobian = new BlockOperator(block_trueOffsets);
         Jacobian->SetBlock(0,0,&Kmat);
         Jacobian->SetBlock(0,2,&Mmat);
         Jacobian->SetBlock(1,1,Mdtpr);
         Jacobian->SetBlock(2,2,Mdtpr);
      }
   }

   return *Jacobian;
}

ReducedSystemOperator::~ReducedSystemOperator()
{
   delete Mdtpr;
   delete ARe;
   delete ASl;
   delete MinvKB;
   delete AReFull;
   delete ScFull;
   delete NbFull;
   delete PwMat;
   delete NbMat;
   delete tmp1;
   delete tmp2;
   delete Jacobian;
   delete Nv;
   delete Nb;
   delete Pw;
   delete vOld;
   delete StabNv;
   delete StabNb;
   delete StabMass;
   delete StabE0;
   delete PB_VPsi; 
   delete PB_VOmega;
   delete PB_BJ;
   delete pd;
   delete bdrForm;
}

void ReducedSystemOperator::Mult(const Vector &k, Vector &y) const
{
   int sc = height/3;

   Vector phiNew(k.GetData() +   0, sc);
   Vector psiNew(k.GetData() +  sc, sc);
   Vector   wNew(k.GetData() +2*sc, sc);

   Vector y1(y.GetData() +   0, sc);
   Vector y2(y.GetData() +  sc, sc);
   Vector y3(y.GetData() +2*sc, sc);

   Vector &k_ = const_cast<Vector &>(k);
   phiGf.MakeTRef(&fespace, k_, 0);
   phiGf.SetFromTrueVector();
   psiGf.MakeTRef(&fespace, k_, sc);
   psiGf.SetFromTrueVector();

   MyCoefficient Bfield(&psiGf, 2);   //we update B
   MyCoefficient velocity(&phiGf, 2);   //we update velocity

   //two different ways to implement Poission Bracket
   //BilinearForm seems a better idea unless we are willing to 
   //sacrifice the accuracy (use a less accurate integrator)
   bool bilinearPB = true;
   if (bilinearPB)
   {
      //------assemble Nv and Nb (operators are assembled locally)------
      delete Nv;
      Nv = new ParBilinearForm(&fespace);
      if (pa)
      {
          Nv->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }
      Nv->AddDomainIntegrator(new ConvectionIntegrator(velocity));
      Nv->Assemble(); 
   }
   else
   {
      //this is not optimized yet
      wGf.MakeTRef(&fespace, k_, 2*sc);
      wGf.SetFromTrueVector();
      delete PB_VPsi;
      PB_VPsi = new ParLinearForm(&fespace);
      PBCoefficient pbCoeff(&phiGf, &psiGf);
      //intOrder = 3*k+0
      PB_VPsi->AddDomainIntegrator(new DomainLFIntegrator(pbCoeff, 3, 0));
      PB_VPsi->Assemble();

      delete PB_VOmega;
      PB_VOmega = new ParLinearForm(&fespace);
      PBCoefficient pbCoeff2(&phiGf, &wGf);
      PB_VOmega->AddDomainIntegrator(new DomainLFIntegrator(pbCoeff2, 3, 0));
      PB_VOmega->Assemble();
   }

   //------compute the current as an auxilary variable (no boundary condition)------
   if (iUpdateJ==0)
   {
      KBMat.Mult(psiNew, z);
      z.Neg();
      M_solver2->Mult(z, J);
   }
   else if (iUpdateJ==1)
   {
      //------compute the current as an auxilary variable (Dirichelt boundary condition)------
      gftmp.SetFromTrueDofs(psiNew);
      Vector Z;
      HypreParMatrix A;
      KB->Mult(gftmp, zFull);
      zFull.Neg(); // z = -z
      M->FormLinearSystem(ess_tdof_list, *j0, zFull, A, J, Z); //apply Dirichelt boundary 
      M_solver->Mult(Z, J); 
   }
   else if (iUpdateJ==2)
   {
      //------compute the current as an auxilary variable (Dirichelt boundary condition)------
      gftmp.SetFromTrueDofs(psiNew);
      Vector Z;
      HypreParMatrix A;
      KB->Mult(gftmp, zFull);
      zFull.Neg(); // z = -z
      Mlumped->FormLinearSystem(ess_tdof_list, *j0, zFull, A, J, Z); //apply Dirichelt boundary 
      M_solver3->Mult(Z, J); 
   }
   else
      MFEM_ABORT("ERROR in ReducedSystemOperator::Mult: wrong option for iUpdateJ"); 

   //+++++compute y1 (this does not have boundary)
   KBMat.Mult(phiNew,y1);
   Mfullmat.Mult(wNew,z);
   y1+=z;

   //+++++compute y3
   add(wNew, -1., *w, zdiff);
   zdiff/=dt;
   //Mmat.Mult(zdiff,y3); //this should not matter due to bchandler
   Mfullmat.Mult(zdiff,y3); 

   if (bilinearPB)
   {
      Nv->TrueAddMult(wNew,y3);
   }
   else
   {
      PB_VOmega->ParallelAssemble(z);
      y3 += z;
   }

   if (DRe!=NULL)
   {
       DRe->TrueAddMult(wNew,y3);
       //bdrForm->TrueAddMult(wNew, y3);
   }

   if (bilinearPB)
   {      
      delete Nb;
      Nb = new ParBilinearForm(&fespace);
      if (pa)
      {
          Nb->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      }

      if (BgradJ==1)
      {
         Nb->AddDomainIntegrator(new ConvectionIntegrator(Bfield));
         Nb->Assemble();
         Nb->TrueAddMult(J, y3, -1.); 
      }
      else if (BgradJ==2)
      {
         Nb->AddDomainIntegrator(new TransposeIntegrator(new ConvectionIntegrator(Bfield)));
         Nb->Assemble();
         Nb->TrueAddMult(J, y3, 1.); 
      }
      else
      {
         Nb->AddDomainIntegrator(new MixedScalarWeakDivergenceIntegrator(Bfield));
         Nb->Assemble();
         Nb->TrueAddMult(J, y3,-1.); 
      }
   }
   else
   {
      //we let J=-J for applying -Nb*J
      J.Neg();
      //use wGf to temporarily hold j
      wGf.MakeTRef(&fespace, J, 0);
      wGf.SetFromTrueVector();
      delete PB_BJ;
      PB_BJ = new ParLinearForm(&fespace);
      PBCoefficient pbCoeff(&psiGf, &wGf);

      PB_BJ->AddDomainIntegrator(new DomainLFIntegrator(pbCoeff, 3, 0));
      PB_BJ->Assemble();
      PB_BJ->ParallelAssemble(z);
      y3 += z;
   }

   //+++++compute y2
   add(psiNew, -1., *psi, z);
   z/=dt;
   //Mmat.Mult(z,y2);//this should not matter due to bchandler
   Mfullmat.Mult(z,y2);
   if (bilinearPB)
   {
      Nv->TrueAddMult(psiNew,y2);
   }
   else
   {
      PB_VPsi->ParallelAssemble(z);
      y2 += z;
   }

   if (DSl!=NULL)
       DSl->TrueAddMult(psiNew,y2);
   if (E0Vec!=NULL)
       y2 += *E0Vec;

   if(usefd || usesupg)
   {
     //first compute an auxilary variable of z3=-∆w (z3=M^-1 KB * w)
     //here z2=(wNew-w)/dt-nu*∆wNew
     KBMat.Mult(wNew, z2);
     M_solver2->Mult(z2, z3);
     add(zdiff, viscosity, z3, z2);
   }

   //compute resiual from y3 to stabilize B.grad Psi
   if(usefd)
   {
     if (true) //XXX turn off for testing adaptive meshes
     {
        delete StabMass;
        StabMass = new ParBilinearForm(&fespace);
        StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, viscosity, Bfield, true));
        StabMass->Assemble(); 
        StabMass->TrueAddMult(z2, y2);

        delete StabNv;
        StabNv = new ParBilinearForm(&fespace);
        StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity, Bfield, true));
        StabNv->Assemble(); 
        StabNv->TrueAddMult(wNew, y2);
     }

     delete StabNb;
     StabNb = new ParBilinearForm(&fespace);
     StabNb->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, Bfield, true));
     StabNb->Assemble(); 
     StabNb->TrueAddMult(J, y2, -1.);
   }

   if(usesupg && im_supg==1)
   {
     //---add supg to y3---
     delete StabMass;
     StabMass = new ParBilinearForm(&fespace);
     StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, viscosity, velocity));
     StabMass->Assemble(); 
     StabMass->TrueAddMult(z2, y3);

     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(wNew, y3);

     /* turn this off for now
     delete StabNb;
     StabNb = new ParBilinearForm(&fespace);
     StabNb->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, Bfield, velocity));
     StabNb->Assemble(); 
     StabNb->TrueAddMult(J, y3, -1.);
     */
   
     KBMat.Mult(psiNew, z2);
     M_solver2->Mult(z2, z3);
     add(z, resistivity, z3, z2);

     //---add supg to y2---
     if(viscosity!=resistivity || itau_!=2)
     {
        if (myid==0 && viscosity!=resistivity ) 
            cout <<"======WARNING: viscosity and resistivity are not identical======"<<endl;
        delete StabMass;
        StabMass = new ParBilinearForm(&fespace);
        StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, resistivity, velocity, itau_));
        StabMass->Assemble(); 

        delete StabNv;
        StabNv = new ParBilinearForm(&fespace);
        StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity, itau_));
        StabNv->Assemble(); 
     }
     StabMass->TrueAddMult(z2, y2);
       StabNv->TrueAddMult(psiNew, y2);

     delete StabE0;
     StabE0 = new ParLinearForm(&fespace);
     StabE0->AddDomainIntegrator(new StabDomainLFIntegrator(dt, resistivity, velocity, *E0rhs, itau_));
     StabE0->Assemble(); 
     StabE0->ParallelAssemble(z);
     y2+=z;
   }
   else if(usesupg && im_supg==2)
   {
     //---add supg only to y3 (omega)---
     delete StabMass;
     StabMass = new ParBilinearForm(&fespace);
     StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, viscosity, velocity));
     StabMass->Assemble(); 
     StabMass->TrueAddMult(z2, y3);

     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(wNew, y3);

     delete StabNb;
     StabNb = new ParBilinearForm(&fespace);
     StabNb->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, Bfield, velocity));
     StabNb->Assemble(); 
     StabNb->TrueAddMult(J, y3, -1.);

     //---add supg to y2---
     if(viscosity!=resistivity || itau_!=2)
     {
        if (myid==0 && viscosity!=resistivity ) 
            cout <<"======WARNING: viscosity and resistivity are not identical======"<<endl;
        delete StabNv;
        StabNv = new ParBilinearForm(&fespace);
        StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity, itau_));
        StabNv->Assemble(); 
     }
     StabNv->TrueAddMult(psiNew, y2);

   }
   else if(usesupg && im_supg==3)
   {
     //this is for testing supg only
     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(wNew, y3);

     if(viscosity!=resistivity)
     {
        delete StabNv;
        StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity));
        StabNv->Assemble(); 
     }
     StabNv->TrueAddMult(psiNew, y2);
   }
   else if(usesupg && im_supg==4)
   {        
     //---add supg only to y2 ---
     KBMat.Mult(psiNew, z2);
     M_solver2->Mult(z2, z3);
     add(z, resistivity, z3, z2);

     //this is for testing supg only
     delete StabMass;
     StabMass = new ParBilinearForm(&fespace);
     StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, resistivity, velocity));
     StabMass->Assemble(); 
     StabMass->TrueAddMult(z2, y2);

     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, resistivity, velocity));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(psiNew, y2);

     delete StabE0;
     StabE0 = new ParLinearForm(&fespace);
     StabE0->AddDomainIntegrator(new StabDomainLFIntegrator(dt, resistivity, velocity, *E0rhs));
     StabE0->Assemble(); 
     StabE0->ParallelAssemble(z);
     y2+=z;
   }
   else if(usesupg && im_supg==5)
   {
     //this only add (v.grad)^2 w and StabMass on w
     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(wNew, y3);

     delete StabMass;
     StabMass = new ParBilinearForm(&fespace);
     StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, viscosity, velocity));
     StabMass->Assemble(); 
     StabMass->TrueAddMult(z2, y3);
   }
   else if(usesupg && im_supg==6)
   {
     //this only add (v.grad)^2 psi
     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(psiNew, y2);
   }
   else if(usesupg && im_supg==7)
   {
     //---add supg to y3 (omega)---
     delete StabMass;
     StabMass = new ParBilinearForm(&fespace);
     StabMass->AddDomainIntegrator(new StabMassIntegrator(dt, viscosity, velocity));
     StabMass->Assemble(); 
     StabMass->TrueAddMult(z2, y3);

     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, velocity));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(wNew, y3);

     delete StabNb;
     StabNb = new ParBilinearForm(&fespace);
     StabNb->AddDomainIntegrator(new StabConvectionIntegrator(dt, viscosity, Bfield, velocity));
     StabNb->Assemble(); 
     StabNb->TrueAddMult(J, y3, -1.);

     //---add special stabilized convection term to y2---
     if (myid==0 && viscosity!=resistivity ) 
         cout <<"======WARNING: viscosity and resistivity are not identical======"<<endl;
     delete StabNv;
     StabNv = new ParBilinearForm(&fespace);
     StabNv->AddDomainIntegrator(new SpecialConvectionIntegrator(dt, resistivity, velocity, itau_));
     StabNv->Assemble(); 
     StabNv->TrueAddMult(psiNew, y2);
   }

   //this step will be done in bchandler anyway
   //y1.SetSubVector(ess_tdof_list, 0.0);
   //y2.SetSubVector(ess_tdof_list, 0.0);
   //y3.SetSubVector(ess_tdof_list, 0.0);
   if (bctype!=1){
       /*
       FunctionCoefficient psiBC(InitialPsi3);
       ParLinearForm rhs2(&fespace);
       rhs2.AddBoundaryIntegrator(new BoundaryLFIntegrator(psiBC));
       rhs2.Assemble();
       rhs2.ParallelAssemble(z);
       */

       bdrForm->TrueAddMult(psiNew, y2); 
       bdrForm->TrueAddMult(wNew, y3); 
   }

   if (false){
       if (myid==0) {
         cout <<"======Debugging: print reisudal as a grid function!!!======"<<endl;
       }
      ostringstream phi_name, psi_name, w_name;
      if (debug)
      {
        phi_name << "dresidual1";// << setfill('0') << setw(6) << myid;
        psi_name << "dresidual2";// << setfill('0') << setw(6) << myid;
          w_name << "dresidual3";// << setfill('0') << setw(6) << myid;
      }
      else
      {
        phi_name << "residual1";//  << setfill('0') << setw(6) << myid;
        psi_name << "residual2";//  << setfill('0') << setw(6) << myid;
         w_name << "residual3" ;// << setfill('0') << setw(6) << myid;
      }

      gftmp.SetFromTrueDofs(y1);
      ofstream osol(phi_name.str().c_str());
      osol.precision(4);
      gftmp.SaveAsOne(osol);

      gftmp.SetFromTrueDofs(y2);
      ofstream osol3(psi_name.str().c_str());
      osol3.precision(4);
      gftmp.SaveAsOne(osol3);

      gftmp.SetFromTrueDofs(y3);
      ofstream osol4(w_name.str().c_str());
      osol4.precision(4);
      gftmp.SaveAsOne(osol4);
   }

   if (debug){
       if (myid==0) {
         cout <<"======Debugging: print reisudal as a grid function!!!======"<<endl;
       }

      /*
      z=0.;
      if (BgradJ==1)
      {
         Nb->TrueAddMult(J, z, -1.); 
      }
      else if (BgradJ==2)
      {
         Nb->TrueAddMult(J, z, 1.); 
      }
      else
      {
         Nb->TrueAddMult(J, z,-1.); 
      }
      */

      gftmp.SetFromTrueDofs(y1);

      /*
      z=0.;
      Mmat.Mult(zdiff,z);
      */
      gftmp2.SetFromTrueDofs(y2);

      gftmp3.SetFromTrueDofs(y3);
      pd->SetCycle(icycle);
      pd->SetTime(icycle);
      icycle++;
      pd->Save();
   }
}

