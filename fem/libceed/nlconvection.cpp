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

#include "nlconvection.hpp"

#include "ceed.hpp"
#ifdef MFEM_USE_CEED
#include "nlconvection.h"
#endif

namespace mfem
{

void CeedPANLConvectionAssemble(const FiniteElementSpace &fes,
                                const mfem::IntegrationRule &irm, CeedData& ceedData)
{
#ifdef MFEM_USE_CEED
   Mesh &mesh = *fes.GetMesh();
   // Perform checks for some assumptions made in the Q-functions.
   MFEM_VERIFY(mesh.Dimension() == mesh.SpaceDimension(), "case not supported");
   MFEM_VERIFY(fes.GetVDim() == mesh.Dimension(), "case not supported");
   int dim = mesh.Dimension();
   CeedPAOperator convOp = {fes, irm,
                            dim * dim, "/nlconvection.h",
                            ":f_build_conv_const", f_build_conv_const,
                            ":f_build_conv_quad", f_build_conv_quad,
                            "", nullptr,
                            "", nullptr,
                            ":f_apply_conv", f_apply_conv,
                            EvalMode::InterpAndGrad,
                            EvalMode::Interp
                           };
   CeedPAAssemble(convOp, ceedData);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

void CeedMFNLConvectionAssemble(const FiniteElementSpace &fes,
                                const mfem::IntegrationRule &irm, CeedData& ceedData)
{
#ifdef MFEM_USE_CEED
   CeedMFOperator convOp = {fes, irm,
                            "/nlconvection.h",
                            ":f_apply_conv_mf_const", f_apply_conv_mf_const,
                            ":f_apply_conv_mf_quad", f_apply_conv_mf_quad,
                            "", nullptr,
                            "", nullptr,
                            EvalMode::InterpAndGrad,
                            EvalMode::Interp
                           };
   CeedMFAssemble(convOp, ceedData);
#else
   mfem_error("MFEM must be built with MFEM_USE_CEED=YES to use libCEED.");
#endif
}

} // namespace mfem