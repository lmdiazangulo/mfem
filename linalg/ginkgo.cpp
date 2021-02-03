// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "../config/config.hpp"

#ifdef MFEM_USE_GINKGO

#include "ginkgo.hpp"
#include "sparsemat.hpp"
#include "../general/globals.hpp"
#include "../general/error.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace mfem
{

namespace GinkgoWrappers
{

GinkgoIterativeSolverBase::GinkgoIterativeSolverBase(
   const std::string &exec_type, int print_iter, int max_num_iter,
   double RTOLERANCE, double ATOLERANCE)
   : Solver(),
     exec_type(exec_type),
     print_lvl(print_iter),
     max_iter(max_num_iter),
     rel_tol(sqrt(RTOLERANCE)),
     abs_tol(sqrt(ATOLERANCE))
{
   if (exec_type == "reference")
   {
      executor = gko::ReferenceExecutor::create();
   }
   else if (exec_type == "omp")
   {
      executor = gko::OmpExecutor::create();
   }
   else if (exec_type == "cuda" && gko::CudaExecutor::get_num_devices() > 0)
   {
      executor = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
   }
   else
   {
      mfem::err <<
                " exec_type needs to be one of the three strings: \"reference\", \"cuda\" or \"omp\" "
                << std::endl;
   }
   using ResidualCriterionFactory = gko::stop::ResidualNormReduction<>;
   residual_criterion             = ResidualCriterionFactory::build()
                                    .with_reduction_factor(rel_tol)
                                    .on(executor);

   combined_factory =
      gko::stop::Combined::build()
      .with_criteria(residual_criterion,
                     gko::stop::Iteration::build()
                     .with_max_iters(max_iter)
                     .on(executor))
      .on(executor);
}

void
GinkgoIterativeSolverBase::initialize_ginkgo_log(gko::matrix::Dense<double>* b)
const
{
   // Add the logger object. See the different masks available in Ginkgo's
   // documentation
   convergence_logger = gko::log::Convergence<>::create(
                           executor, gko::log::Logger::criterion_check_completed_mask);
   residual_logger = std::make_shared<ResidualLogger<>>(executor,
                                                        gko::lend(system_matrix),b);

}

void
GinkgoIterativeSolverBase::Mult(const Vector &x, Vector &y) const
{

   MFEM_VERIFY(system_matrix, "System matrix not initialized");
   MFEM_VERIFY(executor, "executor is not initialized");
   MFEM_VERIFY(y.Size() == x.Size(),
               "Mismatching sizes for rhs and solution");

   using vec       = gko::matrix::Dense<double>;
   if (!iterative_mode)
   {
      y = 0.0;
   }

   // Create x and y vectors in Ginkgo's format. Wrap MFEM's data directly,
   // on CPU or GPU.
   bool on_device = false;
   if (executor != executor->get_master())
   {
      on_device = true;
   }
   auto gko_x = vec::create(executor, gko::dim<2> {x.Size(), 1},
                            gko::Array<double>::view(executor,
                                                     x.Size(), const_cast<double *>(
                                                        x.Read(on_device))), 1);
   auto gko_y = vec::create(executor, gko::dim<2> {y.Size(), 1},
                            gko::Array<double>::view(executor,
                                                     y.Size(), y.ReadWrite(on_device)), 1);

   // Create the logger object to log some data from the solvers to confirm
   // convergence.
   initialize_ginkgo_log(gko::lend(gko_x));

   MFEM_VERIFY(convergence_logger, "convergence logger not initialized" );
   if (print_lvl==1)
   {
      MFEM_VERIFY(residual_logger, "residual logger not initialized" );
      solver_gen->add_logger(residual_logger);
   }

   // Generate the solver from the solver using the system matrix.
   auto solver = solver_gen->generate(system_matrix);

   // Add the convergence logger object to the combined factory to retrieve the
   // solver and other data
   combined_factory->add_logger(convergence_logger);

   // Finally, apply the solver to x and get the solution in y.
   solver->apply(gko::lend(gko_x), gko::lend(gko_y));

   // The convergence_logger object contains the residual vector after the
   // solver has returned. use this vector to compute the residual norm of the
   // solution. Get the residual norm from the logger. As the convergence logger
   // returns a `linop`, it is necessary to convert it to a Dense matrix.
   // Additionally, if the logger is logging on the gpu, it is necessary to copy
   // the data to the host and hence the `residual_norm_d_master`
   auto residual_norm = convergence_logger->get_residual_norm();
   auto residual_norm_d =
      gko::as<gko::matrix::Dense<double>>(residual_norm);
   auto residual_norm_d_master =
      gko::matrix::Dense<double>::create(executor->get_master(),
                                         gko::dim<2> {1, 1});
   residual_norm_d_master->copy_from(residual_norm_d);

   // Get the number of iterations taken to converge to the solution.
   auto num_iteration = convergence_logger->get_num_iterations();

   // Ginkgo works with a relative residual norm through its
   // ResidualNormReduction criterion. Therefore, to get the normalized
   // residual, we divide by the norm of the rhs.
   auto x_norm = gko::matrix::Dense<double>::create(executor->get_master(),
                                                    gko::dim<2> {1, 1});
   if (executor != executor->get_master())
   {
      auto gko_x_cpu = clone(executor->get_master(), gko::lend(gko_x));
      gko_x_cpu->compute_norm2(x_norm.get());
   }
   else
   {
      gko_x->compute_norm2(x_norm.get());
   }

   MFEM_VERIFY(x_norm.get()->at(0, 0) != 0.0, " rhs norm is zero");
   // Some residual norm and convergence print outs. As both
   // `residual_norm_d_master` and `y_norm` are seen as Dense matrices, we use
   // the `at` function to get the first value here. In case of multiple right
   // hand sides, this will need to be modified.
   auto fin_res_norm = std::pow(residual_norm_d_master->at(0,0) / x_norm->at(0,0),
                                2);
   if (num_iteration==max_iter &&
       fin_res_norm > rel_tol )
   {
      converged = 1;
   }
   if (fin_res_norm < rel_tol)
   {
      converged =0;
   }
   if (print_lvl ==1)
   {
      residual_logger->write();
   }
   if (converged!=0)
   {
      mfem::err << "No convergence!" << '\n';
      mfem::out << "(B r_N, r_N) = " << fin_res_norm << '\n'
                << "Number of iterations: " << num_iteration << '\n';
   }
   if (print_lvl >=2 && converged==0 )
   {
      mfem::out << "Converged in " << num_iteration <<
                " iterations with final residual norm "
                << fin_res_norm << '\n';
   }
}

void GinkgoIterativeSolverBase::SetOperator(const Operator &op)
{

   // Only accept SparseMatrix for this type.
   SparseMatrix *op_mat = const_cast<SparseMatrix*>(
                             dynamic_cast<const SparseMatrix*>(&op));
   MFEM_VERIFY(op_mat != NULL,
               "GinkgoIterativeSolverBase::SetOperator : not a SparseMatrix!");
   // Needs to be a square matrix
   MFEM_VERIFY(op_mat->Height() == op_mat->Width(),
               "System matrix is not square");

   bool on_device = false;
   if (executor != executor->get_master())
   {
      on_device = true;
   }

   using mtx = gko::matrix::Csr<double, int>;
   system_matrix = mtx::create(
                      executor, gko::dim<2>(op_mat->Height(), op_mat->Width()),
                      gko::Array<double>::view(executor,
                                               op_mat->NumNonZeroElems(),
                                               op_mat->ReadWriteData(on_device)),
                      gko::Array<int>::view(executor,
                                            op_mat->NumNonZeroElems(),
                                            op_mat->ReadWriteJ(on_device)),
                      gko::Array<int>::view(executor,
                                            op_mat->Height() + 1,
                                            op_mat->ReadWriteI(on_device)));
}

/* ---------------------- CGSolver ------------------------ */
CGSolver::CGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cg = gko::solver::Cg<double>;
   this->solver_gen =
      cg::build().with_criteria(this->combined_factory).on(this->executor);
}

CGSolver::CGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cg         = gko::solver::Cg<double>;
   this->solver_gen = cg::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- BICGSTABSolver ------------------------ */
BICGSTABSolver::BICGSTABSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using bicgstab   = gko::solver::Bicgstab<double>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .on(this->executor);
}

BICGSTABSolver::BICGSTABSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using bicgstab   = gko::solver::Bicgstab<double>;
   this->solver_gen = bicgstab::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- CGSSolver ------------------------ */
CGSSolver::CGSSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cgs = gko::solver::Cgs<double>;
   this->solver_gen =
      cgs::build().with_criteria(this->combined_factory).on(this->executor);
}

CGSSolver::CGSSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using cgs        = gko::solver::Cgs<double>;
   this->solver_gen = cgs::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- FCGSolver ------------------------ */
FCGSolver::FCGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using fcg = gko::solver::Fcg<double>;
   this->solver_gen =
      fcg::build().with_criteria(this->combined_factory).on(this->executor);
}

FCGSolver::FCGSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using fcg        = gko::solver::Fcg<double>;
   this->solver_gen = fcg::build()
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- GMRESSolver ------------------------ */
GMRESSolver::GMRESSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using gmres      = gko::solver::Gmres<double>;
   this->solver_gen = gmres::build()
                      .with_krylov_dim(m)
                      .with_criteria(this->combined_factory)
                      .on(this->executor);
}

GMRESSolver::GMRESSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* preconditioner
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using gmres      = gko::solver::Gmres<double>;
   this->solver_gen = gmres::build()
                      .with_krylov_dim(m)
                      .with_criteria(this->combined_factory)
                      .with_preconditioner(preconditioner)
                      .on(this->executor);
}


/* ---------------------- IRSolver ------------------------ */
IRSolver::IRSolver(
   const std::string &   exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using ir = gko::solver::Ir<double>;
   this->solver_gen =
      ir::build().with_criteria(this->combined_factory).on(this->executor);
}

IRSolver::IRSolver(
   const std::string &exec_type,
   int print_iter,
   int max_num_iter,
   double RTOLERANCE,
   double ATOLERANCE,
   const gko::LinOpFactory* inner_solver
)
   : GinkgoIterativeSolverBase(exec_type, print_iter, max_num_iter, RTOLERANCE,
                               ATOLERANCE)
{
   using ir         = gko::solver::Ir<double>;
   this->solver_gen = ir::build()
                      .with_criteria(this->combined_factory)
                      .with_solver(inner_solver)
                      .on(this->executor);
}


} // namespace GinkgoWrappers

} // namespace mfem

#endif // MFEM_USE_GINKGO
