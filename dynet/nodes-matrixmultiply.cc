#include "dynet/nodes-matrixmultiply.h"

#include "dynet/nodes-macros.h"
#include "dynet/matrix-multiply.h"

using namespace std;

namespace dynet {

// ************* MatrixMultiply *************

#ifndef __CUDACC__

string MatrixMultiply::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " * " << arg_names[1];
  return s.str();
}

Dim MatrixMultiply::dim_forward(const vector<Dim>& xs) const {
  std::cout << xs[0] << " " << xs[1];
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in MatrixMultiply")
  DYNET_ARG_CHECK(xs[0].cols() == xs[1].rows(), "Mismatched input dimensions in MatrixMultiply: " << xs);
  if (xs[1].ndims() == 1) return Dim({xs[0].rows()}, max(xs[0].bd, xs[1].bd));
  return Dim({xs[0].rows(), xs[1].cols()}, max(xs[0].bd, xs[1].bd));
}

int MatrixMultiply::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  // Currently assumes there are two args, and batches with a shared first arg.
  // TODO do we want to treat different dimensions of first/second arg differently?
  if(dim.bd == 1) {
    Sig s(nt::matmul);
    s.add_node(args[0]);
    s.add_dim(cg.nodes[args[1]]->dim);
    return sm.get_idx(s);
  } else {
    return 0; // TODO handle the batched case as well? should it differ at all?
  }
}

std::vector<int> MatrixMultiply::autobatch_concat(const ComputationGraph & cg) const {
  vector<int> ret(args.size(), 0);
  if (dim.bd == 1) { ret[1] = 1; }
  return ret;
}

#endif

template<class MyDevice>
void MatrixMultiply::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in MatrixMultiply::forward");
#ifdef __CUDACC__
  // fx = 0*fx + xs[0] * xs[1]
  dynet::MatrixMultiply(dev, *xs[0], *xs[1], fx, kSCALAR_ZERO);
#else
  DYNET_ASSERT(fx.d.bd == max(xs[0]->d.bd, xs[1]->d.bd), "Failed dimension check in MatrixMultiply::forward");
  if(xs[0]->d.bd == 1) {
    // If the left side has one batch, multiply by columns
    // [x, z, b] = [x, y] * [y, z, b]
    // -> [x, z*b] = [x, y], [y, z*b]
    fx.colbatch_matrix().noalias() = **xs[0] * xs[1]->colbatch_matrix();
  } else {
    // Otherwise, loop over the batches
    DYNET_ASSERT(xs[1]->d.bd == 1 || xs[1]->d.bd == xs[0]->d.bd, "Failed dimension check in MatrixMultiply::forward");
    for(unsigned b = 0; b < xs[0]->d.bd; ++b)
      fx.batch_matrix(b).noalias() = xs[0]->batch_matrix(b) * xs[1]->batch_matrix(b);
  }
#endif
}

template<class MyDevice>
void MatrixMultiply::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in MatrixMultiply::backward");
  int max_b = max(xs[0]->d.bd, xs[1]->d.bd);
#ifdef __CUDACC__
  if (i == 0) {
    if(dEdxi.d.bd == 1 && (dEdf.d.bd == xs[1]->d.bd)) {
      CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
            dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols() * dEdf.d.batch_elems(),
            kSCALAR_ONE,
            dEdf.v, dEdf.d.rows(),
            xs[1]->v, xs[1]->d.rows(),
            kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
    } else {
      for(int b = 0; b < max_b; ++b)
        CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
              dEdxi.d.rows(), dEdxi.d.cols(), dEdf.d.cols(),
              kSCALAR_ONE,
              dEdf.batch_ptr(b), dEdf.d.rows(),
              xs[1]->batch_ptr(b), xs[1]->d.rows(),
              kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
    }
  } else {
    // Do a single multiply if xs[0] has one batch
    if(xs[0]->d.bd == 1) {
      // dEdxi.colbatch_matrix().noalias() += (**xs[0]).transpose() * dEdf.colbatch_matrix();
      CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
            dEdxi.d.rows(), dEdxi.d.cols()*dEdxi.d.batch_elems(), xs[0]->d.rows(),
            kSCALAR_ONE,
            xs[0]->v, xs[0]->d.rows(),
            dEdf.v, dEdf.d.rows(),
            kSCALAR_ONE, dEdxi.v, dEdxi.d.rows()));
    } else {
      for(int b = 0; b < max_b; ++b)
        CUBLAS_CHECK(cublasSgemm(dev.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
              dEdxi.d.rows(), dEdxi.d.cols(), xs[0]->d.rows(),
              kSCALAR_ONE,
              xs[0]->batch_ptr(b), xs[0]->d.rows(),
              dEdf.batch_ptr(b), dEdf.d.rows(),
              kSCALAR_ONE, dEdxi.batch_ptr(b), dEdxi.d.rows()));
    }
  }
#else
  if (i == 0) {
    if(dEdxi.d.bd == 1 && (dEdf.d.bd == xs[1]->d.bd)) {
      (*dEdxi).noalias() += dEdf.colbatch_matrix() * xs[1]->colbatch_matrix().transpose();
    } else {
      for(int b = 0; b < max_b; ++b)
        dEdxi.batch_matrix(b).noalias() += dEdf.batch_matrix(b) * xs[1]->batch_matrix(b).transpose();
    }
  } else {
    if(xs[0]->d.bd == 1) {
      dEdxi.colbatch_matrix().noalias() += (**xs[0]).transpose() * dEdf.colbatch_matrix();
    } else {
      for(int b = 0; b < max_b; ++b)
        dEdxi.batch_matrix(b).noalias() += xs[0]->batch_matrix(b).transpose() * dEdf.batch_matrix(b);
    }
  }
#endif
}
DYNET_NODE_INST_DEV_IMPL(MatrixMultiply)

// ************* TensorContraction *************

#ifndef __CUDACC__

string TensorContraction::as_string(const vector<string>& arg_names) const {
  ostringstream s;
  s << arg_names[0] << " *_tensor_contraction_* " << arg_names[1];
  return s.str();
}

Dim TensorContraction::dim_forward(const vector<Dim>& xs) const {
  std::cout << xs[0] << " " << xs[1];
  DYNET_ARG_CHECK(xs.size() == 2, "Failed input count check in Contraction")
  DYNET_ARG_CHECK(xs[0].d[2] == xs[1].d[0], "Mismatched input dimensions in TensorContraction: " << xs);
  DYNET_ARG_CHECK(xs[0].d[3] == xs[1].d[1], "Mismatched input dimensions in TensorContraction: " << xs);
  return Dim({xs[0].d[0], xs[0].d[1], xs[1].d[2]}, max(xs[0].bd, xs[1].bd));
}

int TensorContraction::autobatch_sig(const ComputationGraph & cg, SigMap &sm) const {
  // Currently assumes there are two args, and batches with a shared first arg.
  // TODO do we want to treat different dimensions of first/second arg differently?
  if(dim.bd == 1) {
    Sig s(nt::tens_contr);
    s.add_node(args[0]);
    s.add_dim(cg.nodes[args[1]]->dim);
    return sm.get_idx(s);
  } else {
    return 0; // TODO handle the batched case as well? should it differ at all?
  }
}

std::vector<int> TensorContraction::autobatch_concat(const ComputationGraph & cg) const {
  vector<int> ret(args.size(), 0);
  if (dim.bd == 1) { ret[1] = 1; }
  return ret;
}

#endif

template<class MyDevice>
void TensorContraction::forward_dev_impl(const MyDevice & dev, const vector<const Tensor*>& xs, Tensor& fx) const {
  DYNET_ASSERT(xs.size() == 2, "Failed dimension check in TensorContraction::forward");
  DYNET_ASSERT(fx.d.bd == max(xs[0]->d.bd, xs[1]->d.bd), "Failed dimension check in TensorContraction::forward");
  //only works for batch size 1 for now
  //only implemented for the case where X[0] is MxNxKxP and X[1] is KxPxR
  //this is the dimension needed to implement the gradient for matrix 
  //multiplication
  int M = xs[0]->d[0];
  int N = xs[0]->d[1];
  int K = xs[0]->d[2];
  int P = xs[0]->d[3];
  int R = xs[1]->d[2];
  if ((K != xs[1]->d[0]) || (P != xs[1]->d[1]) || (M != fx.d[0]) || (N != fx.d[1]) || (R != fx.d[2])) {
    std::cout << "Dimension error in contraction backward!!" << endl;
    return;
  }
  
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int r = 0; r < R; r++) {
        float res = 0.f;
        for (int p = 0; p < P; p++) {
          for (int k = 0; k < K; k++) { 
            res += (xs[0]->tvec()(m*N*K*P+n*K*P+k*P+p)) 
                          * (xs[1]->tvec()(k*P*R + p*R+ r));
          }
          fx.tvec()(m*N*R + n*R + r) = res;
        }
      }
    }
  }

}

template<class MyDevice>
void TensorContraction::backward_dev_impl(const MyDevice & dev,
                             const vector<const Tensor*>& xs,
                             const Tensor& fx,
                             const Tensor& dEdf,
                             unsigned i,
                             Tensor& dEdxi) const {
  DYNET_ASSERT(i < 2, "Failed dimension check in TensorContraction::backward");
  //only implemented for the case where X[0] is MxNxKxP and X[1] is KxPxR
  //this is the dimension needed to implement the gradient for matrix 
  //multiplication
  int M = xs[0]->d[0];
  int N = xs[0]->d[1];
  int K = xs[0]->d[2];
  int P = xs[0]->d[3];
  int R = xs[1]->d[2];
  if ((K != xs[1]->d[0]) || (P != xs[1]->d[1]) || (M != fx.d[0]) || (N != fx.d[1]) || (R != fx.d[2])) {
    std::cout << "Dimension error in contraction backward!!" << endl;
    return;
  }
  if (i == 0) {
    for (int m = 0; m < M; m++) {
      for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
          for (int p = 0; p < P; p++) {
            float res = 0.f;
            for (int r = 0; r < R; r++) {
              res += (dEdf.tvec()(m*N*R + n*R + r)) 
                            * (xs[0]->tvec()(k*P*R + p*R + r));
            }
            dEdxi.tvec()(m*N*K*P + n*K*P + k*P + p) += res;
          }
        }
      }
    }
  } else if (i = 1) {
    for (int k = 0; k < K; k++) {
      for (int p = 0; p < P; p++) {
        for (int r = 0; r < R; r++) {
          float res = 0.f;
          for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
              res+=(dEdf.tvec()(m*N*K*P + n*K*P + k*P + p)) 
                      * (xs[0]->tvec()(m*N*R + n*R + r));
            }
          }
          dEdxi.tvec()(k*P*R + p*R + r) += res;
        }
      }
    }
  }
}
DYNET_NODE_INST_DEV_IMPL(TensorContraction)

}
