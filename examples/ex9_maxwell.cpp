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
//
// Device sample runs:
//    ex9 -pa
//    ex9 -ea
//    ex9 -fa
//    ex9 -pa -m ../data/periodic-cube.mesh
//    ex9 -pa -m ../data/periodic-cube.mesh -d cuda
//    ex9 -ea -m ../data/periodic-cube.mesh -d cuda
//    ex9 -fa -m ../data/periodic-cube.mesh -d cuda
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. The saving of time-dependent data files for external
//               visualization with VisIt (visit.llnl.gov) and ParaView
//               (paraview.org) is also illustrated.

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

    return exp(-40. * (pow(X[0], 2) + pow(X[1], 2)));
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
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

   // 2. Read the mesh from the given mesh file. We can handle geometrically
   //    periodic meshes in this code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();

   // 4. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement, where 'ref_levels' is a
   //    command-line parameter. If the mesh is of NURBS type, we convert it to
   //    a (piecewise-polynomial) high-order mesh.
   for (int lev = 0; lev < ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }
   if (mesh.NURBSext)
   {
      mesh.SetCurvature(max(order, 1));
   }
   mesh.GetBoundingBox(bb_min, bb_max, max(order, 1));

   // 5. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim, BasisType::GaussLobatto);
   FiniteElementSpace fes(&mesh, &fec);
   cout << "Number of unknowns per field:    " << fes.GetVSize() << endl;
   
   // 6. Set up and assemble the bilinear and linear forms corresponding to the
   //    DG discretization. The DGTraceIntegrator involves integrals over mesh
   //    interior faces.
   ConstantCoefficient zero(0.0), one(1.0), mOne(-1.0);
   Vector nxVec(2);  nxVec(0) = 1.0; nxVec(1) = 0.0;
   Vector nyVec(2);  nyVec(0) = 0.0; nyVec(1) = 1.0;
   VectorConstantCoefficient nx(nxVec), ny(nyVec);
         
   BilinearForm MInv(&fes), Kx(&fes), Ky(&fes);
      
   MInv.AddDomainIntegrator(new InverseIntegrator(new MassIntegrator));
   
   double alpha = -1.0, beta = 0.0;
   
   Kx.AddDomainIntegrator(new ConvectionIntegrator(nx));
   Kx.AddInteriorFaceIntegrator(
       new TransposeIntegrator(new DGTraceIntegrator(nx, alpha, beta)));
   Kx.AddBdrFaceIntegrator(
       new TransposeIntegrator(new DGTraceIntegrator(nx, alpha, beta)));
   
   Ky.AddDomainIntegrator(new ConvectionIntegrator(ny));
   Ky.AddInteriorFaceIntegrator(
       new TransposeIntegrator(new DGTraceIntegrator(ny, alpha, beta)));
   Ky.AddBdrFaceIntegrator(
       new TransposeIntegrator(new DGTraceIntegrator(ny, alpha, beta)));
         
   MInv.Assemble();
   int skip_zeros = 0; 
   Kx.Assemble(skip_zeros);
   Ky.Assemble(skip_zeros);
    
   MInv.Finalize();
   Kx.Finalize(skip_zeros);
   Ky.Finalize(skip_zeros);
   
   FunctionCoefficient ez0(u0_function);
   GridFunction ez(&fes), hx(&fes), hy(&fes);
   ez.ProjectCoefficient(ez0);
   hx.ProjectCoefficient(zero);
   hy.ProjectCoefficient(zero);
   
   // Create data collection for solution output: either VisItDataCollection for
   // ascii data files, or SidreDataCollection for binary data files.
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
      pd->SetHighOrderOutput(true);
      pd->SetCycle(0);
      pd->SetTime(0.0);
      pd->Save();
   }

   // 8. Define the time-dependent evolution operator describing the ODE
   //    right-hand side, and perform time-integration (looping over the time
   //    iterations, ti, with a time-step dt).
   double t = 0.0;
   
   Vector aux(fes.GetVSize());
   Vector ezNew(fes.GetVSize()), hxNew(fes.GetVSize()), hyNew(fes.GetVSize());

   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);
      
      Kx.Mult(hy, aux);
      Ky.AddMult(hx, aux, -1.0);
      MInv.Mult(aux, ezNew);
      ezNew *= -dt;
      ezNew.Add(1.0, ez);

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

   // 10. Free the used memory.
   delete pd;

   return 0;
}

