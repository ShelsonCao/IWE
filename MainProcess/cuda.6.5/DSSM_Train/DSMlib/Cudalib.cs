using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace DSMlib
{
    public class Cudalib
    {
        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public extern unsafe static int CudaDeviceCount();

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        [return: MarshalAs(UnmanagedType.BStr)]
        public unsafe static extern string CudaDeviceProperties(int i);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public extern unsafe static int CudaSetDevice(int device);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern IntPtr CudaAllocInt(int e);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern IntPtr CudaAllocFloat(int e);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void CudaDeallocFloat(IntPtr gpu_floats);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void CudaDeallocInt(IntPtr gpu_ints);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void CudaCopyInInt(IntPtr gpu_ints, IntPtr int_array, int len);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void CudaCopyInFloat(IntPtr gpu_floats, IntPtr float_array, int len);


        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void CudaCopyOutFloat(IntPtr gpu_floats, IntPtr float_array, int len);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Zero(IntPtr gpu_floats, int len);



        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Add(IntPtr gpu_floats_a, IntPtr gpu_floats_b, int m, int n, float mweight);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Scale_Matrix(IntPtr gpu_floats_a, int m, int n, float mweight);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Add_Tanh(IntPtr gpu_floats_a, IntPtr gpu_floats_b, int m, int n);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Add_Vector(IntPtr gpu_floats_a, IntPtr gpu_floats_b, int batchsize, int dimension);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Rectified_Vector(IntPtr gpu_floats_a, IntPtr gpu_floats_b, int batchsize, int dimension);


        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Cosine(IntPtr q, IntPtr d, IntPtr dcq, IntPtr dcd, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Cosine_EX(IntPtr q, IntPtr d, IntPtr neg_list, IntPtr dcq, IntPtr dcd, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Derive_Cosine_Linear(IntPtr q, IntPtr d, IntPtr dcq, IntPtr dcd, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Derive_Cosine_Linear_EX(IntPtr q, IntPtr d, IntPtr neg_list, IntPtr dcq, IntPtr dcd, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Derive_Cosine_Rectified(IntPtr q, IntPtr d, IntPtr dcq, IntPtr dcd, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Derive_Cosine_Rectified_EX(IntPtr q, IntPtr d, IntPtr neg_list, IntPtr dcq, IntPtr dcd, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Rectified(IntPtr delta, IntPtr layer_output, int batchsize, int m);

        

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Tanh(IntPtr delta, IntPtr layer_output, int batchsize, int m);

        
        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Multipy(IntPtr delta, IntPtr weight, IntPtr delta_low, int batchsize, int m, int n, int inverse);

        
        
        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Cosine_Similarity(IntPtr a, IntPtr b, IntPtr c, int nTrial, int BATCHSIZE, int mindex, 
									   int batchsize, int dimension, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Cosine_Similarity_EX(IntPtr a,IntPtr b, IntPtr neg_list, IntPtr c, int nTrial, int BATCHSIZE, int mindex, 
									   int batchsize, int dimension, float eps);


        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Calculate_Alpha(IntPtr alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Calculate_Alpha_MXE(IntPtr alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Calculate_Alpha_NCE(IntPtr alpha, IntPtr dist, int nTrial, int BATCHSIZE, int batchsize, float gamma);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Calculate_Alpha_NCE2(IntPtr alpha, IntPtr dist, int nTrial, int BATCHSIZE, int batchsize, float gamma);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void FillOut_Dist_NCE(IntPtr dist, IntPtr neg_list, int nTrial, int BATCHSIZE, int mindex, int batchsize);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Calculate_Alpha_PAIRRANK(IntPtr alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Product(IntPtr a, IntPtr b, IntPtr c, int batchsize, int m, int n);
                        //, int kept, IntPtr alpha, int ntrial, int BATCH_SIZE, int alpha_index);


        

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Convolution_Sparse_Matrix_Product_INTEX(IntPtr deriv, IntPtr maxpooling_index, IntPtr Seg_Index, IntPtr SegMargin_Index, int seg_size, int win_size,
                                        int batchsize, int output_dimension, IntPtr Fea_Index, IntPtr Fea_Value, IntPtr grad, int Feature_Dimension);



        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void SEQ_Sparse_Matrix_Multiply_INTEX(IntPtr Smp_Index, int batchsize, IntPtr Seg_Index, IntPtr Seg_Margin, IntPtr Seg_Len, int seg_size, IntPtr Fea_Index,
                                                   IntPtr Fea_Value, int elementsize, IntPtr mul_weight, IntPtr output, int Feature_dimension, int output_dimension, int win_size);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(IntPtr Smp_Index, int batchsize, IntPtr Seg_Index, IntPtr Seg_Margin, IntPtr Seg_Len, int seg_size, IntPtr Fea_Index,
                                                   IntPtr Fea_Value, int elementsize, IntPtr mul_weight, IntPtr output, int Feature_dimension, int output_dimension, int win_size);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Convolution_Sparse_Matrix_Multiply_INTEX(IntPtr Smp_Index, int batchsize, IntPtr Seg_Index, IntPtr Seg_Margin, IntPtr Seg_Len, int seg_size, IntPtr Fea_Index,
                                                   IntPtr Fea_Value, int elementsize, IntPtr con_weight, IntPtr output, int Feature_dimension, int output_dimension, int win_size);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Max_Pooling(IntPtr pooling_feas, IntPtr Smp_Index, int batchsize, IntPtr output, IntPtr maxpooling_index, int output_dimension);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_WeightAdd(IntPtr gpu_floats_a, IntPtr gpu_floats_b, int batchsize, int dimension, IntPtr mweight, int start, int keep);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_WeightAdd_EX(IntPtr gpu_floats_a, IntPtr gpu_floats_b, IntPtr inver_neg_index, IntPtr inver_neg_value, int batchsize, int dimension, IntPtr mweight, int start, int keep);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Sparse2Dense_Matrix(IntPtr Smp_Idx, IntPtr Fea_Idx, IntPtr Fea_Value, IntPtr matrix, int batchsize, int dimension);
        
        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Aggragate(IntPtr a, IntPtr b, int batchsize, int m);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_Add_OFFSET(IntPtr gpu_floats_a, int offset_a, IntPtr gpu_floats_b, int offset_b, int len, float mweight);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void CUBLAS_Init();
        
        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void  CUBLAS_Destroy();

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern float CUBLAS_Sasum(IntPtr x, int len, int norm);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void CUBLAS_Matrix_Multipy(IntPtr delta, IntPtr weight, IntPtr delta_low, int batchsize, int m, int n, int inverse);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Cosine_Similarity_EX_Full(IntPtr a, IntPtr b, IntPtr neg_list, IntPtr c, int nTrial, int BATCHSIZE, int batchsize, int dimension, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void FillOut_Dist_NCE_Full(IntPtr dist, IntPtr neg_list, int nTrail, int BATCH_SIZE, int batchsize);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Cosine_EX_Full(IntPtr q, IntPtr d, IntPtr neg_list, IntPtr dcq, IntPtr dcd, int nTrail, int BATCHSIZE, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Cosine_Linear_EX_Full(IntPtr q, IntPtr d, IntPtr neg_list, IntPtr dcq, IntPtr dcd, int nTrail, int BATCHSIZE, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Cosine_Rectified_EX_Full(IntPtr q, IntPtr d, IntPtr neg_list, IntPtr dcq, IntPtr dcd, int nTrail, int BATCHSIZE, int batchsize, int m, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_WeightAdd_Full(IntPtr gpu_floats_a, IntPtr gpu_floats_b, int nTrail, int BATCHSIZE, int batchsize, int dimension, IntPtr mweight, int start, int keep);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Matrix_WeightAdd_EX_Full(IntPtr gpu_floats_a, IntPtr gpu_floats_b, IntPtr inver_neg_index, IntPtr inver_neg_value, int nTrial, int BATCHSIZE, int batchsize, int dimension, IntPtr mweight, int start, int keep);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Cosine_Similarity_SubSpace(IntPtr a, IntPtr b, IntPtr c, int labelDim, int BATCHSIZE, int batchsize, int subspaceDim, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void SoftMax(IntPtr a, IntPtr b, int labelDim, int batchsize, float gamma);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_Cosine_Subspace(IntPtr q, IntPtr d, IntPtr dcq, IntPtr dcd, IntPtr alpha, int act_type, int batchsize, int labelDim, int subspaceDim, float gamma, float eps);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void InnerProduct_Similarity(IntPtr a, IntPtr b, IntPtr c, int batchsize, int dimension);

        [DllImport("Cudalib", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern void Deriv_InnerProduct(IntPtr q, IntPtr d, IntPtr dcq, IntPtr dcd, IntPtr alpha, int act_type, int batchsize, int Dim, float gamma, float eps);
    }
}
