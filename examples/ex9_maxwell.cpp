//                                MFEM Example 9
//
// Compile with: make ex9
//
// Sample runs:
//    ex9 -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
//    ex9 -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
//    ex9 -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.002 -tf 9
//    ex9 -m ../data/amr-quad.mesh -p 1 -r 2 -dt 0.02 -s 13 -tf 9
//    ex9 -m ../data/star-q3.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/star-mixed.mesh -p 1 -r 2 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 1 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
//    ex9 -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
//    ex9 -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace std;
using namespace mfem;

// Mesh bounding box
Vector bb_min, bb_max;

// Initial condition
double u0_function(const Vector& x)
{

    // map to the reference [-1,1] domain
    Vector X(2);
    for (size_t i = 0; i < 2; i++) {
        double center = (bb_min[i] + bb_max[i]) * 0.5;
        X[i] = 2 * (x[i] - center) / (bb_max[0] - bb_min[0]);
    }

    return exp(-10. * (pow(X[0], 2) + pow(X[1], 2)));
    //return exp(-10. * pow(X[0], 2));
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/periodic-hexagon.mesh";
   int ref_levels = 2;
   int order = 3;
   const char *device_config = "cpu";
   double t_final = 10.0;
   double dt = 0.01;
   
   bool paraview = false;
   int vis_steps = 5;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                  "--no-paraview-datafiles",
                  "Save data files for ParaView (paraview.org) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   Device device(device_config);
   device.Print();

   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   cout << "Number of unknowns per field:    " << fes.GetVSize() << endl;
   
   ConstantCoefficient zero(0.0), one(1.0), mOne(-1.0);
   Vector nxVec(2);  nxVec(0) = 1.0; nxVec(1) = 0.0;
   Vector nyVec(2);  nyVec(0) = 0.0; nyVec(1) = 1.0;
   Vector n1Vec(2);  n1Vec(0) = 1.0; n1Vec(1) = 1.0;
   VectorConstantCoefficient nx(nxVec), ny(nyVec), n1(n1Vec);

   BilinearForm MInv(&fes), Kx(&fes) , Ky(&fes);

   MInv.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   
   double alpha = -1.0, beta = 0.0;
   
   Kx.AddDomainIntegrator(new DerivativeIntegrator(one, 0));
   Kx.AddInteriorFaceIntegrator(
       new TransposeIntegrator(new DGTraceIntegrator(nx, alpha, beta)));
   
   Ky.AddDomainIntegrator(new DerivativeIntegrator(one, 1));
   Ky.AddInteriorFaceIntegrator(
       new TransposeIntegrator(new DGTraceIntegrator(ny, alpha, beta)));
         
   MInv.Assemble();
   int skip_zeros = 0; 
   Kx.Assemble(skip_zeros);
   Ky.Assemble(skip_zeros);
    
   MInv.Finalize();
   Kx.Finalize(skip_zeros);
   Ky.Finalize(skip_zeros);

   FunctionCoefficient u0(u0_function);
   GridFunction ez(&fes), hx(&fes), hy(&fes);
   ez.ProjectCoefficient(u0);
   hx.ProjectCoefficient(zero);
   hy.ProjectCoefficient(zero);

   ParaViewDataCollection *pd = NULL;
   if (paraview)
   {
      pd = new ParaViewDataCollection("Example9", &mesh);
      pd->SetPrefixPath("ParaView");
      pd->RegisterField("ez", &ez);
      pd->RegisterField("hx", &hx);
      pd->RegisterField("hy", &hy);
      pd->SetLevelsOfDetail(order);
      pd->SetDataFormat(VTKFormat::BINARY);
      order > 0 ? pd->SetHighOrderOutput(true) : pd->SetHighOrderOutput(false);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   double t = 0.0;
   
   Vector aux(fes.GetVSize());
   Vector ezNew(fes.GetVSize()), hxNew(fes.GetVSize()), hyNew(fes.GetVSize());
   
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      
      
      // Update E.
      Kx.Mult(hy, aux);
      Ky.AddMult(hx, aux, -1.0);
      MInv.Mult(aux, ezNew);
      ezNew *= -dt;
      ezNew.Add(1.0, ez);


      // Update H.
      Kx.Mult(ezNew, aux);
      MInv.Mult(aux, hyNew);
      hyNew *= -dt;
      hyNew.Add(1.0, hy);

      Ky.Mult(ezNew, aux);
      MInv.Mult(aux, hxNew);
      hxNew *= dt;
      hxNew.Add(1.0, hx);

      ez = ezNew;
      hx = hxNew;
      hy = hyNew;

      t += dt;
      ti++;

      done = (t >= t_final - 1e-8*dt);

      if (done || ti % vis_steps == 0)
      {
         cout << "time step: " << ti << ", time: " << t << endl;

         if (paraview)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
      }
   }

   delete pd;

   return 0;
}

