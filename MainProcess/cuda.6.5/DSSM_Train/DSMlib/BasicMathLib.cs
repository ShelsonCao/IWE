using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DSMlib
{
    public class BasicMathlib
    {
        public static int THREAD_NUMBER = ParameterSetting.BasicMathLibThreadNum;
        
        public static void SEQ_Sparse_Matrix_Multiply_INTEX(int[] Smp_Index, int batchsize, int[] Seg_Index, int[] Seg_Margin, float[] Seg_Len, int seg_size, int[] Fea_Index, float[] Fea_Value, int elementsize, float[] mul_weight, float[] output, int inputDimension, int outputDimension, int winSize)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * outputDimension;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int batch_idx = id / outputDimension;
                        int output_idx = id % outputDimension;

                        int seg_end = Smp_Index[batch_idx];
                        int seg_begin = 0;
                        if (batch_idx > 0)
                        {
                            seg_begin = Smp_Index[batch_idx - 1];
                        }

                        float sum = 0;
                        for (int word_idx = seg_begin; word_idx < seg_end; ++word_idx)
                        {
                            int col_end = Seg_Index[word_idx];
                            int col_begin = 0;
                            if (word_idx > 0)
                            {
                                col_begin = Seg_Index[word_idx - 1];
                            }
                            for (int i = col_begin; i < col_end; ++i)
                            {
                                int fea_idx = Fea_Index[i];
                                sum += Fea_Value[i] * mul_weight[((word_idx - seg_begin) * inputDimension + fea_idx) * outputDimension + output_idx];
                            }
                        }
                        output[batch_idx * outputDimension + output_idx] = sum;
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }
        public static void Convolution_Sparse_Matrix_Multiply_INTEX(int[] Smp_Index, int batchsize, int[] Seg_Index, int[] Seg_Margin, float[] Seg_Len, int seg_size, int[] Fea_Index, float[] Fea_Value, int elementsize, float[] mul_weight, float[] output, int inputDimension, int outputDimension, int winSize)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = outputDimension * seg_size;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int idx = id / seg_size;
                        int idy = id % seg_size;

                        output[idy * outputDimension + idx] = 0;
                        int ws = winSize / 2;
                        int mSmp_idx = Seg_Margin[idy];
                        float sum = 0;

                        for (int w = -ws; w <= ws; w++)
                        {
                            if (idy + w >= 0 && idy + w < seg_size)
                            {
                                if (Seg_Margin[idy + w] == mSmp_idx)
                                {
                                    float mlen = 1; //Seg_Len[idy+w]; // sqrtf(Seg_Len[idy+w]);
                                    int row = idy + w; // idx / n;
                                    int col_end = Seg_Index[row];
                                    int col_begin = 0;
                                    if (row > 0)
                                    {
                                        col_begin = Seg_Index[row - 1];
                                    }

                                    for (int i = col_begin; i < col_end; i++)
                                    {
                                        int fea_idx = Fea_Index[i];
                                        if (fea_idx >= inputDimension)
                                        {
                                            continue;
                                        }
                                        sum += Fea_Value[i] * 1.0f / mlen * mul_weight[((w + ws) * inputDimension + fea_idx) * outputDimension + idx];
                                    }
                                }
                            }
                        }
                        output[idy * outputDimension + idx] = sum;
                    }
                }
            });
        }
        public static void Max_Pooling(float[] pooling_feas, int[] Smp_Index, int batchsize, float[] output, int[] maxpooling_index, int output_dimension)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = output_dimension * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int batch_idx = id / output_dimension;
                        int output_idx = id % output_dimension;
                        output[batch_idx * output_dimension + output_idx] = 0;
                        int col_end = Smp_Index[batch_idx];
                        int col_begin = 0;
                        if (batch_idx > 0)
                        {
                            col_begin = Smp_Index[batch_idx - 1];
                        }
                        float max_value = 0;
                        int max_index = -1;
                        for (int i = col_begin; i < col_end; i++)
                        {
                            if (max_index == -1 || pooling_feas[i * output_dimension + output_idx] > max_value)
                            {
                                max_value = pooling_feas[i * output_dimension + output_idx];
                                max_index = i;
                            }
                        }
                        output[batch_idx * output_dimension + output_idx] = max_value;
                        maxpooling_index[batch_idx * output_dimension + output_idx] = max_index;
                    }
                    else
                    {
                        break;
                    }
                }

            });
        }

        public static void Matrix_Multipy(float[] input, float[] weight, float[] output, int batchsize, int inputDimension, int outputDimension, int inverse)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = outputDimension * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int idy = id / outputDimension;
                        int idx = id % outputDimension;

                        int row = idy; // / n;
                        int col = idx; // % n;
                        float sum = 0;
                        for (int i = 0; i < inputDimension; i++)
                        {
                            if (inverse == 1)
                            {
                                sum += input[row * inputDimension + i] * weight[col * inputDimension + i];
                            }
                            else
                            {
                                sum += input[row * inputDimension + i] * weight[i * outputDimension + col];
                            }
                        }
                        output[idy * outputDimension + idx] = sum;
                    }
                    else
                    {
                        break;
                    }
                }
            });

        }

        public static void Matrix_Add_Tanh(float[] output, float[] bias, int batchsize, int output_number)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * output_number;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int batch_idx = id / output_number;
                        int output_idx = id % output_number;

                        float m = output[batch_idx * output_number + output_idx] + bias[output_idx];
                        output[batch_idx * output_number + output_idx] = (float)(Math.Tanh(m));
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Matrix_Add_Vector(float[] output, float[] bias, int batchsize, int output_number)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * output_number;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int batch_idx = id / output_number;
                        int output_idx = id % output_number;

                        float m = output[batch_idx * output_number + output_idx] + bias[output_idx];
                        output[batch_idx * output_number + output_idx] = m;
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Matrix_Rectified_Vector(float[] output, float[] bias, int batchsize, int output_number)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * output_number;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int batch_idx = id / output_number;
                        int output_idx = id % output_number;

                        float m = output[batch_idx * output_number + output_idx] + bias[output_idx];
                        if (m < 0)
                        {
                            m = 0;
                        }
                        output[batch_idx * output_number + output_idx] = m;
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Cosine_Similarity(float[] a, float[] b, float[] c, int nTrialPlus1, int BATCH_SIZE, int mindex, int batchsize, int dimension, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int process_len = (batchsize + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < batchsize)
                    {
                        float sumxx = 0;
                        float sumyy = 0;
                        float sumxy = 0;
                        for (int i = 0; i < dimension; i++)
                        {
                            sumxx += a[idx * dimension + i] * a[idx * dimension + i];
                            sumyy += b[idx * dimension + i] * b[idx * dimension + i];
                            sumxy += a[idx * dimension + i] * b[idx * dimension + i];
                        }
                        c[mindex * BATCH_SIZE + idx] = (float)(sumxy * 1.0f / (Math.Sqrt((float)(sumxx * sumyy)) + eps));
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Cosine_Similarity_EX(float[] a, float[] b, int[] neg_list, float[] c, int nTrialPlus1, int BATCH_SIZE, int mindex, int batchsize, int dimension, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int process_len = (batchsize + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < batchsize)
                    {
                        float sumxx = 0;
                        float sumyy = 0;
                        float sumxy = 0;
                        int mtindex = neg_list[idx];
                        for (int i = 0; i < dimension; i++)
                        {
                            sumxx += a[idx * dimension + i] * a[idx * dimension + i];
                            sumyy += b[mtindex * dimension + i] * b[mtindex * dimension + i];
                            sumxy += a[idx * dimension + i] * b[mtindex * dimension + i];
                        }
                        c[mindex * BATCH_SIZE + idx] = (float)(sumxy * 1.0f / (Math.Sqrt((float)(sumxx * sumyy)) + eps));
                    }

                }
            });
        }

        static void cal_alpha(float[] alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = (nTrial - 1) * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        int row = idx / batchsize;
                        int col = idx % batchsize;
                        alpha[row * BATCHSIZE + col + BATCHSIZE] = (float)Math.Exp((float)(-gamma * (alpha[col] - alpha[row * BATCHSIZE + col + BATCHSIZE])));
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }


        static void cal_alpha_sum(float[] alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma, int init)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float sum = init;
                        for (int i = 1; i < nTrial; i++)
                        {
                            sum += alpha[i * BATCHSIZE + idx];
                        }
                        alpha[idx] = sum;
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        static void cal_alpha_norm(float[] alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = (nTrial - 1) * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        int row = idx / batchsize;
                        int col = idx % batchsize;
                        alpha[row * BATCHSIZE + col + BATCHSIZE] = (float)((gamma * alpha[row * BATCHSIZE + col + BATCHSIZE]) / alpha[col]);
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        static void cal_alpha_norm_MXE(float[] alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = (nTrial - 1) * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        int row = idx / batchsize;
                        int col = idx % batchsize;
                        alpha[row * BATCHSIZE + col + BATCHSIZE] = (float)((gamma * alpha[row * BATCHSIZE + col + BATCHSIZE]) / alpha[col] / alpha[col]);
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Calculate_Alpha(float[] alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            cal_alpha(alpha, nTrial, BATCHSIZE, batchsize, gamma);
            cal_alpha_sum(alpha, nTrial, BATCHSIZE, batchsize, gamma, 1);
            cal_alpha_norm(alpha, nTrial, BATCHSIZE, batchsize, gamma);
            cal_alpha_sum(alpha, nTrial, BATCHSIZE, batchsize, gamma, 0);
        }

        public static void Calculate_Alpha_MXE(float[] alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            cal_alpha(alpha, nTrial, BATCHSIZE, batchsize, gamma);
            cal_alpha_sum(alpha, nTrial, BATCHSIZE, batchsize, gamma, 1);
            cal_alpha_norm_MXE(alpha, nTrial, BATCHSIZE, batchsize, gamma);
            cal_alpha_sum(alpha, nTrial, BATCHSIZE, batchsize, gamma, 0);
        }


        static void cal_alpha_nce(float [] alpha, float [] dist, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = nTrial * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        int row = idx / batchsize;
                        int col = idx % batchsize;
                        alpha[row * BATCHSIZE + col] = (float)(gamma / (1.0f + (nTrial - 1) * Math.Exp(dist[row * BATCHSIZE + col] - gamma * alpha[row * BATCHSIZE + col] + gamma))); //+gamma is from hxd, sd doesn't have this
                        
                        if (idx < batchsize) alpha[idx] = gamma - alpha[idx];
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Calculate_Alpha_NCE(float[] alpha, float[] dist, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            cal_alpha_nce(alpha, dist, nTrial, BATCHSIZE, batchsize, gamma);
        }

        static void cal_alpha_nce2(float[] alpha, float[] dist, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = nTrial * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        int row = idx / batchsize;
                        int col = idx % batchsize;
                        float s = (float)(1.0f / (1.0f + (nTrial - 1) * Math.Exp(dist[row * BATCHSIZE + col] - gamma * alpha[row * BATCHSIZE + col] + gamma))); //+gamma is from hxd, sd doesn't have this
                        alpha[row * BATCHSIZE + col] = gamma * s * (1.0f - s);
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Calculate_Alpha_NCE2(float[] alpha, float[] dist, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            cal_alpha_nce2(alpha, dist, nTrial, BATCHSIZE, batchsize, gamma);
        }
        
        public static void Calculate_Alpha_PAIRRANK(float[] alpha, int nTrial, int BATCHSIZE, int batchsize, float gamma)
        {
            //if (idx < batchsize)
            //{
            //    float msum = 0;
            //    for (int n = 1; n < nTrial; n++)
            //    {
            //        float a = gamma * (1.0f - 1.0f / (1 + expf(-gamma * (alpha[idx] - alpha[n * BATCHSIZE + idx]))));
            //        alpha[n * BATCHSIZE + idx] = a;
            //        msum += a;
            //    }
            //    alpha[idx] = msum;
            //}

            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float msum = 0;
                        for (int n = 1; n < nTrial; n++)
                        {
                            float a = (float)(gamma * (1.0f - 1.0f / (1 + Math.Exp(-gamma * (alpha[idx] - alpha[n * BATCHSIZE + idx])))));
                            alpha[n * BATCHSIZE + idx] = a;
                            msum += a;
                        }
                        alpha[idx] = msum;
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void FillOut_Dist_NCE(float[] dist, int[] neg_list, int nTrailPlus1, int BATCH_SIZE, int mindex, int batchsize)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int process_len = (batchsize + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < batchsize)
                    {
                        int mtindex = neg_list[idx];
                        dist[mindex * BATCH_SIZE + idx] = dist[mtindex];
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Deriv_Cosine(float[] q, float[] d, float[] dcq, float[] dcd, int batchsize, int m, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float a = 0;
                        float b = eps;
                        float c = eps;
                        for (int i = 0; i < m; i++)
                        {
                            a += q[idx * m + i] * d[idx * m + i];
                            b += q[idx * m + i] * q[idx * m + i];
                            c += d[idx * m + i] * d[idx * m + i];
                        }
                        b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
                        for (int i = 0; i < m; i++)
                        {
                            dcq[idx * m + i] = (float)((1 - q[idx * m + i]) * (1 + q[idx * m + i]) * (d[idx * m + i] * 1.0f / (b * c) - q[idx * m + i] * a * 1.0f / (b * b * b * c)));
                            dcd[idx * m + i] = (float)((1 - d[idx * m + i]) * (1 + d[idx * m + i]) * (q[idx * m + i] * 1.0f / (b * c) - d[idx * m + i] * a * 1.0f / (b * c * c * c)));
                            dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
                            dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Derive_Cosine_Linear(float[] q, float[] d, float[] dcq, float[] dcd, int batchsize, int m, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float a = 0;
                        float b = eps;
                        float c = eps;
                        for (int i = 0; i < m; i++)
                        {
                            a += q[idx * m + i] * d[idx * m + i];
                            b += q[idx * m + i] * q[idx * m + i];
                            c += d[idx * m + i] * d[idx * m + i];
                        }
                        b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
                        for (int i = 0; i < m; i++)
                        {
                            dcq[idx * m + i] = (float)((d[idx * m + i] * 1.0f / (b * c) - q[idx * m + i] * a * 1.0f / (b * b * b * c)));
                            dcd[idx * m + i] = (float)((q[idx * m + i] * 1.0f / (b * c) - d[idx * m + i] * a * 1.0f / (b * c * c * c)));
                            dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
                            dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
                        }

                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Derive_Cosine_Rectified(float[] q, float[] d, float[] dcq, float[] dcd, int batchsize, int m, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float a = 0;
                        float b = eps;
                        float c = eps;
                        for (int i = 0; i < m; i++)
                        {
                            a += q[idx * m + i] * d[idx * m + i];
                            b += q[idx * m + i] * q[idx * m + i];
                            c += d[idx * m + i] * d[idx * m + i];
                        }
                        b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
                        for (int i = 0; i < m; i++)
                        {
                            if (q[idx * m + i] == 0)    //TODO: float ==
                            {
                                dcq[idx * m + i] = 0;
                            }
                            else
                            {
                                dcq[idx * m + i] = (float)((d[idx * m + i] * 1.0f / (b * c) - q[idx * m + i] * a * 1.0f / (b * b * b * c)));
                            }
                            dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;

                            if (d[idx * m + i] == 0)    //TODO: float ==
                            {
                                dcd[idx * m + i] = 0;
                            }
                            else
                            {
                                dcd[idx * m + i] = (float)((q[idx * m + i] * 1.0f / (b * c) - d[idx * m + i] * a * 1.0f / (b * c * c * c)));
                            }
                            dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
                        }


                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Deriv_Cosine_EX(float[] q, float[] d, int[] neg_list, float[] dcq, float[] dcd, int batchsize, int m, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float a = 0;
                        float b = eps;
                        float c = eps;
                        int mIndex = neg_list[idx];
                        for (int i = 0; i < m; i++)
                        {
                            a += q[idx * m + i] * d[mIndex * m + i];
                            b += q[idx * m + i] * q[idx * m + i];
                            c += d[mIndex * m + i] * d[mIndex * m + i];
                        }
                        b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
                        for (int i = 0; i < m; i++)
                        {
                            dcq[idx * m + i] = (float)((1 - q[idx * m + i]) * (1 + q[idx * m + i]) * (d[mIndex * m + i] * 1.0f / (b * c) - q[idx * m + i] * a * 1.0f / (b * b * b * c)));
                            dcd[idx * m + i] = (float)((1 - d[mIndex * m + i]) * (1 + d[mIndex * m + i]) * (q[idx * m + i] * 1.0f / (b * c) - d[mIndex * m + i] * a * 1.0f / (b * c * c * c)));
                            dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
                            dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
                        }

                    }
                    else
                    {
                        break;
                    }
                }
            });
        }
        public static void Derive_Cosine_Linear_EX(float[] q, float[] d, int[] neg_list, float[] dcq, float[] dcd, int batchsize, int m, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        {
                            float a = 0;
                            float b = eps;
                            float c = eps;
                            int mIndex = neg_list[idx];
                            for (int i = 0; i < m; i++)
                            {
                                a += q[idx * m + i] * d[mIndex * m + i];
                                b += q[idx * m + i] * q[idx * m + i];
                                c += d[mIndex * m + i] * d[mIndex * m + i];
                            }
                            b = (float)Math.Sqrt(b);
                            c = (float)Math.Sqrt(c);
                            for (int i = 0; i < m; i++)
                            {
                                dcq[idx * m + i] = (float)((d[mIndex * m + i] * 1.0f / (b * c) - q[idx * m + i] * a * 1.0f / (b * b * b * c)));
                                dcd[idx * m + i] = (float)((q[idx * m + i] * 1.0f / (b * c) - d[mIndex * m + i] * a * 1.0f / (b * c * c * c)));
                                dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;
                                dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
                            }
                        }

                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Derive_Cosine_Rectified_EX(float[] q, float[] d, int[] neg_list, float[] dcq, float[] dcd, int batchsize, int m, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float a = 0;
                        float b = eps;
                        float c = eps;
                        int mIndex = neg_list[idx];
                        for (int i = 0; i < m; i++)
                        {
                            a += q[idx * m + i] * d[mIndex * m + i];
                            b += q[idx * m + i] * q[idx * m + i];
                            c += d[mIndex * m + i] * d[mIndex * m + i];
                        }
                        b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
                        for (int i = 0; i < m; i++)
                        {
                            if (q[idx * m + i] == 0)    //TODO: float ==
                            {
                                dcq[idx * m + i] = 0;
                            }
                            else
                            {
                                dcq[idx * m + i] = (float)((d[mIndex * m + i] * 1.0f / (b * c) - q[idx * m + i] * a * 1.0f / (b * b * b * c)));
                            }
                            dcq[idx * m + i] = dcq[idx * m + i] * 1.0f / batchsize;


                            if (d[mIndex * m + i] == 0) //TODO: float ==
                            {
                                dcd[idx * m + i] = 0;
                            }
                            else
                            {
                                dcd[idx * m + i] = (float)((q[idx * m + i] * 1.0f / (b * c) - d[mIndex * m + i] * a * 1.0f / (b * c * c * c)));
                            }
                            dcd[idx * m + i] = dcd[idx * m + i] * 1.0f / batchsize;
                        }


                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Matrix_WeightAdd(float[] gpu_floats_a, float[] gpu_floats_b, int batchsize, int dimension, float[] mweight, int start, int keep)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * dimension;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int idy = id / dimension;
                        int idx = id % dimension;
                        if (keep != 0)
                        {
                            gpu_floats_a[idy * dimension + idx] += keep * gpu_floats_b[idy * dimension + idx] * mweight[start + idy];
                        }
                        else
                        {
                            gpu_floats_a[idy * dimension + idx] = gpu_floats_b[idy * dimension + idx] * mweight[start + idy];
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Matrix_WeightAdd_EX(float[] gpu_floats_a, float[] gpu_floats_b, int[] inver_neg_index, int[] inver_neg_value, int batchsize, int dimension, float[] mweight, int start, int keep)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * dimension;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int idy = id / dimension;
                        int idx = id % dimension;
                        int col_end = inver_neg_index[idy];
                        int col_begin = 0;
                        if (idy > 0)
                        {
                            col_begin = inver_neg_index[idy - 1];
                        }
                        float sum = 0;
                        for (int i = col_begin; i < col_end; i++)
                        {
                            int row = inver_neg_value[i];
                            sum += gpu_floats_b[row * dimension + idx] * mweight[start + row];
                        }
                        if (keep != 0)
                        {
                            gpu_floats_a[idy * dimension + idx] += keep * sum;
                        }
                        else
                        {
                            gpu_floats_a[idy * dimension + idx] = sum;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Deriv_Tanh(float[] delta, float[] layer_output, int batchsize, int m)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * m;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int idy = id / m;
                        int idx = id % m;
                        delta[idy * m + idx] = delta[idy * m + idx] * (1 - layer_output[idy * m + idx]) * (1 + layer_output[idy * m + idx]);
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }


        public static void Deriv_Rectified(float[] delta, float[] layer_output, int batchsize, int m)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * m;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < total)
                    {
                        int idy = id / m;
                        int idx = id % m;
                        if (layer_output[idy * m + idx] == 0)   // TODO: Float ==
                        {
                            delta[idy * m + idx] = 0; // delta[idy * m +idx] ;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }	
		//


        public static void SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(int[] Smp_Index, int batchsize, int[] Seg_Index, int[] Seg_Margin, float[] Seg_Len,
                                                              int seg_size, int[] Fea_Index,
                                                   float[] Fea_Value, int elementsize,
                                                   float[] mul_weight, float[] output, int Feature_dimension, int output_dimension, int win_size)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = output_dimension;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        int seg_begin = 0;
                        for (int sample = 0; sample < batchsize; ++sample)
                        {
                            int seg_end = Smp_Index[sample];
                            for (int word_idx = seg_begin; word_idx < seg_end; ++word_idx)
                            {
                                int col_end = Seg_Index[word_idx];
                                int col_begin = 0;
                                if (word_idx > 0)
                                {
                                    col_begin = Seg_Index[word_idx - 1];
                                }
                                for (int i = col_begin; i < col_end; ++i)
                                {
                                    int fea_idx = Fea_Index[i];

                                    mul_weight[((word_idx - seg_begin) * Feature_dimension + fea_idx) * output_dimension + idx] += Fea_Value[i] * output[sample * output_dimension + idx];
                                }
                            }
                            seg_begin = seg_end;
                        }
                    }

                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Convolution_Sparse_Matrix_Product_INTEX(float[] deriv, int[] maxpooling_index, int[] Seg_Index, int[] SegMargin_Index, int seg_size, int win_size,
                                        int batchsize, int output_dimension, int[] Fea_Index, float[] Fea_Value, float[] grad, int Feature_Dimension)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = output_dimension * win_size;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        int output_idx = idx / win_size;
                        int win_idx = idx % win_size;

                        //float sum = 0;
                        for (int b = 0; b < batchsize; b++)
                        {
                            int target_seg = maxpooling_index[b * output_dimension + output_idx];

                            if (target_seg == -1)
                            {
                                continue;
                            }
                            int target_smp = SegMargin_Index[target_seg];
                            //deriv[i * output_dimension + idx] *  
                            int ws = win_size / 2;
                            int w = win_idx - ws;
                            int row = target_seg + w; // idx / n;
                            if (row >= 0 && row < seg_size)
                            {
                                if (SegMargin_Index[row] == target_smp)
                                {
                                    int col_end = Seg_Index[row];
                                    int col_begin = 0;
                                    if (row > 0)
                                    {
                                        col_begin = Seg_Index[row - 1];
                                    }
                                    //float sum = 0;
                                    for (int i = col_begin; i < col_end; i++)
                                    {
                                        int fea_idx = Fea_Index[i];
                                        if (fea_idx >= Feature_Dimension)
                                        {
                                            continue;
                                        }
                                        float m = Fea_Value[i] * deriv[b * output_dimension + output_idx];
                                        // con_weight[((w+ws) * Feature_dimension + fea_idx)*output_dimension+idx];
                                        grad[(win_idx * Feature_Dimension + fea_idx) * output_dimension + output_idx] += m;
                                    }
                                }
                            }
                        }

                    }

                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Matrix_Product(float[] a, float[] b, float[] c, int batchsize, int m, int n)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = n * m;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        int idx = cnt / m;
                        int idy = cnt % m;
                        int row = idy; // / n;
                        int col = idx;// % n;
                        float sum = 0;
                        for (int i = 0; i < batchsize; i++)
                        {
                            sum += a[i * m + row] * b[i * n + col]; // * alpha[alpha_index * BATCH_SIZE + i]; // vAlpha(i);
                        }
                        //if(kept == 0)
                        //{
                        c[idy * n + idx] = sum;
                        //}
                        //else
                        //{
                        //	c[idy * n + idx] += sum;
                        //}
                    }
                    else
                    {
                        break;
                    }
                }
            });

        }

        public static void Matrix_Add(float[] gpu_floats_a, float[] gpu_floats_b, int m, int n, float mweight)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = n * m;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        int idx = cnt / m;
                        int idy = cnt % m;
                        gpu_floats_a[idy * n + idx] = gpu_floats_a[idy * n + idx] + gpu_floats_b[idy * n + idx] * mweight;
                    }
                    else
                    {
                        break;
                    }
                }
            });

        }

        public static float VectorInnerProduct(float[] a, int offset_a, float[] b, int offset_b, int len)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int process_len = (len + THREAD_NUM - 1) / THREAD_NUM;

            float[] tmp = new float[THREAD_NUM];
            Array.Clear(tmp, 0, THREAD_NUM);

            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                tmp[thread_idx] = 0;
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < len)
                    {
                        tmp[thread_idx] += a[offset_a + cnt] * b[offset_b + cnt];
                    }
                    else
                    {
                        break;
                    }
                }
            });

            float sum = 0;
            for (int i = 0; i < THREAD_NUM; i++)
            {
                sum += tmp[i];
            }
            return sum;
        }

        public static void Scale_Matrix(float[] gpu_floats_a, int m, int n, float mweight)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = n * m;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        int idx = cnt / m;
                        int idy = cnt % m;
                        gpu_floats_a[idy * n + idx] = gpu_floats_a[idy * n + idx] * mweight; //(float)log( (float)gpu_floats_a[idx]);
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Matrix_Aggragate(float[] a, float[] b, int batchsize, int m)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = m;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        float sum = 0;
                        for (int i = 0; i < batchsize; i++)
                        {
                            sum += a[i * m + cnt]; //* alpha[alpha_index * BATCH_SIZE + i];
                        }
                        b[cnt] = sum;
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Sparse2Dense_Matrix(int[] Smp_Idx, int[] Fea_Idx, float[] Fea_Value, float[] matrix, int batchsize, int outputDimension)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        int end = Smp_Idx[cnt];
                        int begin = cnt >= 1 ? Smp_Idx[cnt - 1] : 0;
                        for (int k = begin; k < end; k++)
                        {
                            matrix[cnt * outputDimension + Fea_Idx[k]] = Fea_Value[k];
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }
		
		public static void Cosine_Similarity_EX_Full(float[] a, float[] b, int[] neg_list, float[] c, int nTrial, int BATCHSIZE, int batchsize, int dimension, float eps)
		{
			int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * nTrial;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
						int idy = cnt / batchsize;
                        int idx = cnt % batchsize;
						
						float sumxx = 0;
						float sumyy = 0;
						float sumxy = 0;
						//float * a_iter = a + (idx * dimension);
						//float * b_iter = b + (neg_list[idy * batchsize + idx] * dimension);
						//float * a_iter_end = a_iter + dimension;
						
						int nid = neg_list[idy * batchsize + idx] * dimension;
						int pid = idx * dimension;
						for(int i=0;i<dimension;i++)
						{
							sumxx += a[pid + i] * a[pid + i];
							sumyy += b[nid + i] * b[nid + i];
							sumxy += a[pid + i] * b[nid + i];
						}
						c[ (idy + 1) * BATCHSIZE + idx] = (float)( sumxy / ((float)Math.Sqrt(sumxx * sumyy) + eps) );
					}
                    else
                    {
                        break;
                    }
                }
            });
		}
		
		public static void FillOut_Dist_NCE_Full(float[] dist, int[] neg_list, int nTrail, int BATCH_SIZE, int batchsize)
		{
			int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * nTrail;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
						int idy = cnt / batchsize;
                        int idx = cnt % batchsize;
						int mtindex = neg_list[idy * BATCH_SIZE + idx];
                        dist[BATCH_SIZE + idy * BATCH_SIZE + idx] = dist[mtindex];
					}
                    else
                    {
                        break;
                    }
                }
            });
		}
		
		public static void Deriv_Cosine_EX_Full(float[] q, float[] d, int[] neg_list, float[] dcq, float[] dcd, int nTrail, int BATCHSIZE, int batchsize, int m, float eps)
		{
			int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * nTrail;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
						int idy = cnt / batchsize;
                        int idx = cnt % batchsize;
						
						float a = 0;
						float b = 0;
						float c = 0;
						float bc, a_bbbc, a_bccc, batchsizenorm;
		
						int qid = idx * m;
						int did = neg_list[idy * BATCHSIZE + idx] * m;
						for(int i=0;i<m;i++)
						{
							b += q[qid + i] * q[qid + i];
							c += d[did + i] * d[did + i];
							a += q[qid + i] * d[did + i];
						}
		
						b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
						bc = b*c + eps;
						a_bbbc = a/(b*b*b*c + eps);
						a_bccc = a/(b*c*c*c + eps);

						batchsizenorm = 1.0f / batchsize;
						
						int dc_qd_id = idy * (BATCHSIZE * m) + idx * m;
						for(int i=0;i<m;i++)
						{
							dcq[dc_qd_id + i] = (1.0f - q[qid + i]) * ( 1.0f + q[qid + i]) * (d[did + i] / bc - q[qid + i] * a_bbbc) * batchsizenorm;
							dcd[dc_qd_id + i] = (1.0f -d[did + i]) * ( 1.0f + d[did + i]) * (q[qid + i] / bc - d[did + i] * a_bccc) * batchsizenorm;
						}
					}
                    else
                    {
                        break;
                    }
                }
            });
		}
		
		public static void Deriv_Cosine_Linear_EX_Full(float[] q, float[] d, int[] neg_list, float[] dcq, float[] dcd, int nTrail, int BATCHSIZE, int batchsize,  int m, float eps)
		{
			int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * nTrail;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
						int idy = cnt / batchsize;
                        int idx = cnt % batchsize;
						
						float a = 0;
						float b = eps;
						float c = eps;
						int mIndex = neg_list[idy * BATCHSIZE + idx];
						for(int i=0;i<m;i++)
						{
							a += q[idx * m + i] * d[mIndex * m + i];
							b += q[idx * m + i] * q[idx * m + i];
							c += d[mIndex * m + i] * d[mIndex * m + i];
						}
						b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
		
						for(int i=0;i<m;i++)
						{
							dcq[idy * BATCHSIZE * m  + idx * m + i] = (float)( (d[mIndex*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
							dcd[idy * BATCHSIZE * m  + idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[mIndex*m+i] * a * 1.0f / (b*c*c*c)) );
							dcq[idy * BATCHSIZE * m  + idx * m + i] = dcq[idy * BATCHSIZE * m  + idx * m + i] * 1.0f / batchsize;
							dcd[idy * BATCHSIZE * m  + idx * m + i] = dcd[idy * BATCHSIZE * m  + idx * m + i] * 1.0f / batchsize;
						}
		
					}
                    else
                    {
                        break;
                    }
                }
            });
		}

		public static void Deriv_Cosine_Rectified_EX_Full(float[] q, float[] d, int[] neg_list, float[] dcq, float[] dcd, int nTrail, int BATCHSIZE, int batchsize,  int m, float eps)
		{
			int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * nTrail;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
						int idy = cnt / batchsize;
                        int idx = cnt % batchsize;
						
						float a = 0;
						float b = eps;
						float c = eps;
						int mIndex = neg_list[idy * BATCHSIZE + idx];
						for(int i=0;i<m;i++)
						{
							a += q[idx * m + i] * d[mIndex * m + i];
							b += q[idx * m + i] * q[idx * m + i];
							c += d[mIndex * m + i] * d[mIndex * m + i];
						}
						b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);
		
						for(int i=0;i<m;i++)
						{
							if(q[idx*m+i] == 0)
							{
								dcq[idy * BATCHSIZE * m + idx * m + i] = 0;
							}
							else
							{
								dcq[idy * BATCHSIZE * m + idx * m + i] = (float)( (d[mIndex*m+i] * 1.0f / (b*c) - q[idx*m+i] * a * 1.0f / (b*b*b*c)) );
							}
							dcq[idy * BATCHSIZE * m + idx * m + i] = dcq[idy * BATCHSIZE * m + idx * m + i] * 1.0f / batchsize;

							if(d[mIndex*m+i] == 0)
							{
								dcd[idy * BATCHSIZE * m + idx * m + i] = 0;
							}
							else
							{
								dcd[idy * BATCHSIZE * m + idx * m + i] = (float)( (q[idx*m+i] * 1.0f / (b*c) - d[mIndex*m+i] * a * 1.0f / (b*c*c*c)) );
							}
							dcd[idy * BATCHSIZE * m + idx * m + i] = dcd[idy * BATCHSIZE * m + idx * m + i] * 1.0f / batchsize;
						}
					}
                    else
                    {
                        break;
                    }
                }
            });
		}
		
		public static void Matrix_WeightAdd_Full(float[] gpu_floats_a, float[] gpu_floats_b, int nTrail, int BATCHSIZE, int batchsize, int dimension, float[] mweight, int start, int keep)
		{
			int THREAD_NUM = THREAD_NUMBER;
            int total = dimension * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
						int idx = cnt / batchsize;
						int idy = cnt % batchsize;
						for(int i=0;i<nTrail;i++)
						{
							gpu_floats_a[idy*dimension+idx] += keep * gpu_floats_b[ i * BATCHSIZE * dimension + idy * dimension + idx] * mweight[start + i * BATCHSIZE + idy];
						}
					}
                    else
                    {
                        break;
                    }
                }
            });
		}
		
		public static void Matrix_WeightAdd_EX_Full(float[] gpu_floats_a, float[] gpu_floats_b, 
				int[] inver_neg_index, int[] inver_neg_value, int nTrial, int BATCHSIZE, int batchsize, int dimension, float[] mweight, int start, int keep)
		{
			int THREAD_NUM = THREAD_NUMBER;
            int total = dimension * batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
						int idx = cnt / batchsize;
						int idy = cnt % batchsize;
						
						for(int n=0; n<nTrial; n++)
						{
							int col_end = inver_neg_index[n * BATCHSIZE + idy];
							int col_begin = 0;
							if(idy > 0)
							{
								col_begin = inver_neg_index[n * BATCHSIZE + idy - 1];
							}

							float sum = 0;
							for(int i=col_begin; i<col_end; i++)
							{
								int row = inver_neg_value[n * BATCHSIZE + i];
								sum += gpu_floats_b[n * BATCHSIZE * dimension + row * dimension + idx] * mweight[start + n * BATCHSIZE + row];
							}

							gpu_floats_a[idy*dimension+idx] += keep * sum;
						}
					}
                    else
                    {
                        break;
                    }
                }
            });
		}


        public static void Cosine_Similarity_SubSpace(float[] a, float[] b, float[] c, int labelDim, int BATCHSIZE, int batchsize, int subspaceDim, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * labelDim;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        int idx = cnt / labelDim;
                        int idy = cnt % labelDim;
                        float sumxx = 0;
                        float sumyy = 0;
                        float sumxy = 0;
                        int id_start = idx * (labelDim * subspaceDim) + idy * subspaceDim;
                        for (int i = 0; i < subspaceDim; i++)
                        {
                            sumxx += a[id_start + i] * a[id_start + i];
                            sumyy += b[id_start + i] * b[id_start + i];
                            sumxy += a[id_start + i] * b[id_start + i];
                        }
                        c[idx * labelDim + idy] = (float)(sumxy * 1.0f / (Math.Sqrt((float)(sumxx * sumyy)) + eps));
                    }
                    else
                    {
                        break;
                    }
                }
            });

            
        }

        public static void SoftMax(float[] a, float[] b, int labelDim, int batchsize, float gamma)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float log_sum = 0;

                        for (int i = 0; i < labelDim; i++)
                        {
                            float tmpa = gamma * a[idx * labelDim + i];
                            if (i == 0)
                            {
                                log_sum = tmpa;
                                continue;
                            }
                            else
                            {
                                if (log_sum >= tmpa)
                                {
                                    log_sum = log_sum + (float)Math.Log(1 + Math.Exp(gamma * (tmpa - log_sum)));
                                }
                                else
                                {
                                    log_sum = tmpa + (float)Math.Log(1 + Math.Exp(gamma * (log_sum - tmpa)));
                                }
                            }
                        }

                        for (int i = 0; i < labelDim; i++)
                        {
                            float tmpa = gamma * a[idx * labelDim + i];
                            b[idx * labelDim + i] = (float)Math.Exp(tmpa - log_sum);
                        }
					}
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Deriv_Cosine_Subspace(float[] q, float[] d, float[] dcq, float[] dcd, float[] alpha, int act_type, int batchsize, int labelDim, int subspaceDim, float gamma, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize * labelDim;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        int idx = cnt / labelDim;
                        int idy = cnt % labelDim;

                        float alpha_v = gamma * alpha[idx * labelDim + idy];
                        int id_start = idx * labelDim * subspaceDim + idy * subspaceDim;
                        float a = 0;
                        float b = eps;
                        float c = eps;
                        for (int i = 0; i < subspaceDim; i++)
                        {
                            a += q[id_start + i] * d[id_start + i];
                            b += q[id_start + i] * q[id_start + i];
                            c += d[id_start + i] * d[id_start + i];
                        }
                        b = (float)Math.Sqrt(b);
                        c = (float)Math.Sqrt(c);

                        /// tanh function.
                        if (act_type == 0)
                        {
                            for (int i = 0; i < subspaceDim; i++)
                            {
                                dcq[id_start + i] = (float)((1 - q[id_start + i]) * (1 + q[id_start + i]) * (d[id_start + i] * 1.0f / (b * c) - q[id_start + i] * a * 1.0f / (b * b * b * c)));
                                dcd[id_start + i] = (float)((1 - d[id_start + i]) * (1 + d[id_start + i]) * (q[id_start + i] * 1.0f / (b * c) - d[id_start + i] * a * 1.0f / (b * c * c * c)));
                                dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;
                                dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
                            }
                        }
                        /// linear function.
                        else if (act_type == 1)
                        {
                            for (int i = 0; i < subspaceDim; i++)
                            {
                                dcq[id_start + i] = (float)((d[id_start + i] * 1.0f / (b * c) - q[id_start + i] * a * 1.0f / (b * b * b * c)));
                                dcd[id_start + i] = (float)((q[id_start + i] * 1.0f / (b * c) - d[id_start + i] * a * 1.0f / (b * c * c * c)));
                                dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;
                                dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
                            }
                        }
                        /// 
                        else if (act_type == 2)
                        {
                            for (int i = 0; i < subspaceDim; i++)
                            {
                                if (Math.Abs(q[id_start + i]) < eps)
                                {
                                    dcq[id_start + i] = 0;
                                }
                                else
                                {
                                    dcq[id_start + i] = (float)((d[id_start + i] * 1.0f / (b * c) - q[id_start + i] * a * 1.0f / (b * b * b * c)));
                                }
                                dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;

                                if (Math.Abs(d[id_start + i]) < eps)
                                {
                                    dcd[id_start + i] = 0;
                                }
                                else
                                {
                                    dcd[id_start + i] = (float)((q[id_start + i] * 1.0f / (b * c) - d[id_start + i] * a * 1.0f / (b * c * c * c)));
                                }
                                dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
                            }
                        }
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void InnerProduct_Similarity(float[] a, float[] b, float[] c, int batchsize, int dimension)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
                        float sumxy = 0;
                        for (int i = 0; i < dimension; i++)
                        {
                            sumxy += a[idx * dimension + i] * b[idx * dimension + i];
                        }
                        c[idx] = (float)(sumxy * 1.0f);
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Deriv_InnerProduct(float[] q, float[] d, float[] dcq, float[] dcd, float[] alpha, int act_type, int batchsize, int Dim, float gamma, float eps)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = batchsize;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int idx = thread_idx * process_len + t;
                    if (idx < total)
                    {
						
						float alpha_v = gamma * alpha[idx];
		                int id_start = idx * Dim;

		                /// tanh function.
		                if(act_type == 0)
		                {
			                for(int i=0;i<Dim;i++)
			                {
				                dcq[id_start + i] = (float)( (1 - q[id_start + i]) * ( 1 + q[id_start + i]) * d[id_start + i] * alpha_v * 1.0f );
				                dcd[id_start + i] = (float)( (1 - d[id_start + i]) * ( 1 + d[id_start + i]) * q[id_start + i] * alpha_v * 1.0f );
				                //dcq[id_start + i] = alpha_v * dcq[id_start + i] ;
				                //dcd[id_start + i] = alpha_v * dcd[id_start + i] ;
			                }
		                }
		                /// linear function.
		                else if(act_type == 1)
		                {
			                for(int i=0;i<Dim;i++)
			                {
				                dcq[id_start + i] = (float)( d[id_start + i] * alpha_v * 1.0f  );
				                dcd[id_start + i] = (float)( q[id_start + i] * alpha_v * 1.0f  );
				                // dcq[id_start + i] = alpha_v * dcq[id_start + i] * 1.0f / batchsize;
				                // dcd[id_start + i] = alpha_v * dcd[id_start + i] * 1.0f / batchsize;
			                }
		                }
		                /// 
		                else if(act_type == 2)
		                {
			                for(int i=0;i<Dim;i++)
			                {
				                if(Math.Abs(q[id_start + i]) < eps)
				                {
					                dcq[id_start + i]  = 0;
				                }
				                else
				                {
					                dcq[id_start + i] = (float)( d[id_start + i] * alpha_v * 1.0f  );
				                }
				
			
				                if(Math.Abs(d[id_start + i]) < eps)
				                {
					                dcd[id_start + i ] =0;
				                }
				                else
				                {
					                dcd[id_start + i] = (float)( q[id_start + i] * alpha_v * 1.0f  );
				                }
				
			                }
		                }
					}
                    else
                    {
                        break;
                    }
                }
            });
        }

        public static void Matrix_Add_OFFSET(float[] a, int offset_a, float[] b, int offset_b, int len, float mweight)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int total = len;
            int process_len = (total + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.For(0, THREAD_NUM, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int cnt = thread_idx * process_len + t;
                    if (cnt < total)
                    {
                        a[offset_a + cnt] += b[offset_b + cnt] * mweight; //* alpha[alpha_index * BATCH_SIZE + i];
                    }
                    else
                    {
                        break;
                    }
                }
            });
        }
	}
}