using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text.RegularExpressions;
using jlib;
using Util;

namespace sent2vec
{
    public enum A_Func { Linear = 0, Tanh = 1, Rectified = 2 };
    public enum N_Type { Fully_Connected = 0, Convolution_layer = 1 };
    public enum P_Pooling { MAX_Pooling = 0 };
    public class NeuralLayer
    {
        public int Number;
        public NeuralLayer(int num)
        {
            Number = num;
        }
    }

    public class NeuralLink
    {
        public NeuralLayer Neural_In;
        public NeuralLayer Neural_Out;

        public float[] Back_Weight;
        public float[] Back_Bias;

        public N_Type Nt = N_Type.Fully_Connected;
        public int N_Winsize = 1;
        public P_Pooling pool_type = P_Pooling.MAX_Pooling;
        public A_Func Af;
        public float isHidBias = 0;
        public NeuralLink(NeuralLayer layer_in, NeuralLayer layer_out, A_Func af)
        {
            Neural_In = layer_in;
            Neural_Out = layer_out;

            Back_Weight = new float[Neural_In.Number * Neural_Out.Number];
            Back_Bias = new float[Neural_Out.Number];

            Af = af;
        }

        public NeuralLink(NeuralLayer layer_in, NeuralLayer layer_out, A_Func af,  float isBias, N_Type nt, int win_size)
        {
            Neural_In = layer_in;
            Neural_Out = layer_out;
            Nt = nt;
            N_Winsize = win_size;

            Af = af;
            isHidBias = isBias;

            Back_Weight = new float[Neural_In.Number * Neural_Out.Number * N_Winsize];
            Back_Bias = new float[Neural_Out.Number];

        }

    }
    public class Letter_Feature_Extraction
    {
        public int MAX_TOKEN_NUM = 20; //
        public int NormalizeFeatValueLengthLimit = int.MaxValue;

        List<Dictionary<int, float>> NormalizeFeatureValues(List<Dictionary<int, float>> rawInput)
        {
            float maxValue = rawInput.Max(o => o.Max(kvp => kvp.Value));
            if (Math.Abs(maxValue) < 1e-8)
            {
                return rawInput;
            }
            List<Dictionary<int, float>> ret = new List<Dictionary<int, float>>();
            foreach (var dict in rawInput)
            {
                var retDict = new Dictionary<int, float>();
                foreach (var kvp in dict)
                {
                    retDict.Add(kvp.Key, kvp.Value / maxValue);
                }
                ret.Add(retDict);
            }
            return ret;
        }

        Dictionary<int, float> NormalizeFeatureValues(Dictionary<int, float> rawInput)
        {
            return NormalizeFeatureValues(new List<Dictionary<int, float>>() { rawInput }).First();
        }

        public Dictionary<string, int> vocab_dict = null;
        internal Dictionary<string, int> FillVocab_Size(string fileName)
        {
            Dictionary<string, int> vocab_list = new System.Collections.Generic.Dictionary<string, int>();
            string[] lines = File.ReadAllLines(fileName, Encoding.UTF8);
            for (int i = 0; i < lines.Length; i++)
            {
                string[] items = lines[i].Split('\t');
                if (items[0].Length >= 3)
                {
                    vocab_list.Add(items[0], vocab_list.Count);
                }
            }
            return vocab_list;
        }

        public static string[] SplitWords(string input)
        {
            string[] words = input.Split(new string[] { " ", "\t", "\n", "\r" }, StringSplitOptions.RemoveEmptyEntries).ToArray();
            return words;
        }

        public Dictionary<int, float> ToLetterNGramString(int ngram, string[] words, int pos)
        {
            Dictionary<int, float> feas = new Dictionary<int, float>();

            foreach (string word in words)
            {
                string mword = "#" + word + "#";
                for (int i = 0; i < mword.Length - ngram + 1; i++)
                {
                    string letterNGram = mword.Substring(i, ngram);
                    if (vocab_dict.ContainsKey(letterNGram))
                    {
                        int vindex = vocab_dict[letterNGram];
                        if (feas.ContainsKey(vindex))
                        {
                            feas[vindex+pos] = feas[vindex] + 1;
                        }
                        else
                        {
                            feas[vindex+pos] = 1;
                        }
                    }
                }
            }
            return feas;
        }

        public void LoadVocab(string fileName)
        {
            vocab_dict = FillVocab_Size(fileName);
        }

        public Dictionary<int, float> Feature_Extract_BOW(string strstream, Dictionary<string, Dictionary<int, float>> dic)
        {
            Dictionary<int, float> ret = new Dictionary<int, float>();
            string[] words = SplitWords(strstream);

            if (words.Length > 1)
            {
                throw new Exception("DSSM cannot support FeatureType of 'we' when there are more than one word!");
            }
            
            string word = words[0];
            if (dic.ContainsKey(word))
            {
                ret = dic[word];
            }

            return ret;
        }

        public Dictionary<int, float> Feature_Extract_BOW(string strstream, int pos)
        {
            string[] words = SplitWords(strstream);
            var ret = ToLetterNGramString(3, words, pos);
            if (words.Length > NormalizeFeatValueLengthLimit)
            {
                ret = NormalizeFeatureValues(ret);
            }
            return ret;
        }

        public List<Dictionary<int, float>> Feature_Extractor_SOW(string str, Dictionary<string, Dictionary<int, float>> dic)
        {
            List<Dictionary<int, float>> mdicts = new List<Dictionary<int, float>>();
            if (str.Trim().Equals(""))
            {
                return mdicts;
            }
            if (str.Length >= 10000000)
            {
                return mdicts;
            }
            string[] words = SplitWords(str);

            foreach (string word in words)
            {
                if (dic.ContainsKey(word))
                {
                    mdicts.Add(dic[word]);
                }
            }

            return mdicts;
        }

        public List<Dictionary<int, float>> Feature_Extractor_SOW(string str)
        {
            List<Dictionary<int, float>> mdicts = new List<Dictionary<int, float>>(); // Dictionary<int, int>();
            if (str.Trim().Equals(""))
            {
                return mdicts;
            }
            if (str.Length >= 10000000)
            {
                return mdicts;
            }
            string[] words = SplitWords(str);

            Dictionary<int, float> tmp_p = null;
            foreach (string word in words)
            {
                if (mdicts.Count < MAX_TOKEN_NUM)
                {
                    tmp_p = new Dictionary<int, float>();
                }

                string mword = "#" + word + "#";
                for (int ngram = 3; ngram <= 3; ngram++)
                {
                    for (int i = 0; i < mword.Length - ngram + 1; i++)
                    {
                        string nletter = mword.Substring(i, ngram);
                        if (vocab_dict.ContainsKey(nletter))
                        {
                            int mkey = vocab_dict[nletter];
                            if (tmp_p.ContainsKey(mkey))
                            {
                                tmp_p[mkey] += 1;
                            }
                            else
                            {
                                tmp_p.Add(mkey, 1);
                            }
                        }
                    }
                }

                if (tmp_p.Count > 0 && mdicts.Count < MAX_TOKEN_NUM)
                {
                    mdicts.Add(tmp_p);
                }
            }
            if (words.Length > NormalizeFeatValueLengthLimit)
            {
                mdicts = NormalizeFeatureValues(mdicts);
            }
            return mdicts;
        }
    }

    public class BasicMathlib
    {
        public static int THREAD_NUMBER = CmpInfo.ProcessorCount;
        public static float Cos_Similarity(float[] x, float[] y, int size)
        {
            float sumxy = 0;
            float sumx = 0;
            float sumy = 0;
            for (int i = 0; i < size; i++)
            {
                sumx += x[i] * x[i];
                sumy += y[i] * y[i];
                sumxy += x[i] * y[i];
            }
            return (float)(sumxy * 1.0 / Math.Sqrt(sumx * sumy + float.Epsilon));
        }

        public static float Cos_Similarity(List<float> x, List<float> y)
        {
            if (x.Count != y.Count)
            {
                Console.WriteLine("Cosine Similarity : The size of two column is not matched");
                return 0;
            }
            float sumxy = 0;
            float sumx = 0;
            float sumy = 0;
            for (int i = 0; i < x.Count; i++)
            {
                sumx += x[i] * x[i];
                sumy += y[i] * y[i];
                sumxy += x[i] * y[i];
            }
            return (float)(sumxy * 1.0 / Math.Sqrt(sumx * sumy + float.Epsilon));
        }

        public static void Sparse_Matrix_Multiply_INTEX(int[] sample, int[] fea_idx, float[] fea_value, int elementsize,
                            float[] Weight, float[] output, int batchsize, int output_num, int input_num, int output_offset)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int[] thread_num = new int[THREAD_NUM];
            for (int i = 0; i < THREAD_NUM; i++)
            {
                thread_num[i] = i;
            }
            int process_len = (batchsize * output_num + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.ForEach(thread_num, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < batchsize * output_num)
                    {
                        int batch_idx = id / output_num;
                        int output_idx = id % output_num;
                        int col_end = sample[batch_idx];
                        int col_begin = 0;
                        if (batch_idx > 0)
                            col_begin = sample[batch_idx - 1];

                        float sum = 0;
                        for (int i = col_begin; i < col_end; i++)
                        {
                            int mfea = fea_idx[i];
                            if (mfea >= input_num)
                                continue;
                            sum += fea_value[i] * Weight[mfea * output_num + output_idx];
                        }
                        output[batch_idx * output_num + output_idx + output_offset] = sum;
                    }
                }
            });
        }

        public static void Matrix_Multiply(float[] a, float[] b, float[] c, int batchsize, int output_number, int input_number, int input_offset, int output_offset)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int[] thread_num = new int[THREAD_NUM];
            for (int i = 0; i < THREAD_NUM; i++)
            {
                thread_num[i] = i;
            }
            int process_len = (batchsize * output_number + THREAD_NUM - 1) / THREAD_NUM;

            Parallel.ForEach(thread_num, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < batchsize * output_number)
                    {
                        int batch_idx = id / output_number;
                        int output_idx = id % output_number;

                        float sum = 0;
                        for (int i = 0; i < input_number; i++)
                        {
                            sum += a[batch_idx * input_number + i + input_offset] * b[i * output_number + output_idx];
                        }
                        c[batch_idx * output_number + output_idx + output_offset] = sum;
                    }
                }
            });
        }

        public static void Matrix_Add_Tanh(float[] output, float[] bias, int batchsize, int output_number)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int[] thread_num = new int[THREAD_NUM];
            for (int i = 0; i < THREAD_NUM; i++)
            {
                thread_num[i] = i;
            }
            int process_len = (batchsize * output_number + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.ForEach(thread_num, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < batchsize * output_number)
                    {
                        int batch_idx = id / output_number;
                        int output_idx = id % output_number;

                        float m = output[batch_idx * output_number + output_idx]; // +bias[output_idx];
                        output[batch_idx * output_number + output_idx] = (float)(Math.Tanh(m));

                    }
                }
            });
        }

        public static void Convolution_Sparse_Matrix_Multiply_INTEX(int[] Smp_Index, int batchsize, int[] Seg_Index, int[] Seg_Margin, int seg_size, int[] Fea_Index,
                                                   float[] Fea_Value, int elementsize,
                                                   float[] con_weight, float[] output, int Feature_dimension, int output_dimension, int win_size)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int[] thread_num = new int[THREAD_NUM];
            for (int i = 0; i < THREAD_NUM; i++)
            {
                thread_num[i] = i;
            }
            int process_len = (output_dimension * seg_size + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.ForEach(thread_num, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < output_dimension * seg_size)
                    {
                        int seg_idx = id / output_dimension;
                        int output_idx = id % output_dimension;

                        output[seg_idx * output_dimension + output_idx] = 0;
                        int ws = win_size / 2;
                        int mSmp_idx = Seg_Margin[seg_idx];
                        float sum = 0;
                        for (int w = -ws; w <= ws; w++)
                        {
                            if (seg_idx + w >= 0 && seg_idx + w < seg_size)
                            {
                                if (Seg_Margin[seg_idx + w] == mSmp_idx)
                                {
                                    float mlen = 1; //Seg_Len[idy+w]; // sqrtf(Seg_Len[idy+w]);
                                    int row = seg_idx + w; // idx / n;
                                    int col_end = Seg_Index[row];
                                    int col_begin = 0;
                                    if (row > 0)
                                    {
                                        col_begin = Seg_Index[row - 1];
                                    }

                                    for (int i = col_begin; i < col_end; i++)
                                    {
                                        int fea_idx = Fea_Index[i];
                                        if (fea_idx >= Feature_dimension)
                                            continue;
                                        sum += Fea_Value[i] * 1.0f / mlen * con_weight[((w + ws) * Feature_dimension + fea_idx) * output_dimension + output_idx];
                                    }
                                }
                            }
                        }
                        output[seg_idx * output_dimension + output_idx] = sum;
                    }
                }
            });
        }

        public static void Max_Pooling(float[] pooling_feas, int[] Smp_Index, int batchsize, float[] output, int[] maxpooling_index, int output_dimension, int output_offset)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int[] thread_num = new int[THREAD_NUM];
            for (int i = 0; i < THREAD_NUM; i++)
            {
                thread_num[i] = i;
            }
            int process_len = (output_dimension * batchsize + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.ForEach(thread_num, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < output_dimension * batchsize)
                    {
                        int batch_idx = id / output_dimension;
                        int output_idx = id % output_dimension;
                        output[batch_idx * output_dimension + output_idx + output_offset] = 0;
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
                        output[batch_idx * output_dimension + output_idx + output_offset] = max_value;
                        maxpooling_index[batch_idx * output_dimension + output_idx] = max_index;
                    }
                }

            });
        }

        public static void Cosine_Similarity(float[] output_query, float[] output_doc, float[] alpha, int batchsize, int output_dim)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int[] thread_num = new int[THREAD_NUM];
            for (int i = 0; i < THREAD_NUM; i++)
            {
                thread_num[i] = i;
            }
            int process_len = (batchsize + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.ForEach(thread_num, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < batchsize)
                    {
                        int batch_idx = id;
                        float sumxx = 0;
                        float sumyy = 0;
                        float sumxy = 0;
                        for (int i = 0; i < output_dim; i++)
                        {
                            sumxx += output_query[batch_idx * output_dim + i] * output_query[batch_idx * output_dim + i];
                            sumyy += output_doc[batch_idx * output_dim + i] * output_doc[batch_idx * output_dim + i];
                            sumxy += output_query[batch_idx * output_dim + i] * output_doc[batch_idx * output_dim + i];

                        }
                        alpha[batch_idx] = (float)(sumxy * 1.0f / (Math.Sqrt((float)(sumxx * sumyy)) + float.Epsilon));
                    }
                }
            });
        }

        /// <summary>
        /// Assuming batch size = 1 now.
        /// </summary>
        /// <param name="p1"></param>
        /// <param name="batchsize"></param>
        /// <param name="p2"></param>
        /// <param name="p3"></param>
        internal static void Max_PoolingDense(float[] pooling_feas, int batchsize, float[] output, int[] maxpooling_index, int pool_size, int output_dimension)
        {
            int THREAD_NUM = THREAD_NUMBER;
            int[] thread_num = new int[THREAD_NUM];
            for (int i = 0; i < THREAD_NUM; i++)
            {
                thread_num[i] = i;
            }
            int process_len = (output_dimension * batchsize + THREAD_NUM - 1) / THREAD_NUM;
            Parallel.ForEach(thread_num, thread_idx =>
            {
                for (int t = 0; t < process_len; t++)
                {
                    int id = thread_idx * process_len + t;
                    if (id < output_dimension * batchsize)
                    {
                        int batch_idx = id / output_dimension;
                        int output_idx = id % output_dimension;
                        output[batch_idx * output_dimension + output_idx] = 0;
                        int col_end = pool_size;
                        int col_begin = 0;                        
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
                        output[batch_idx * output_dimension + output_idx ] = max_value;
                        maxpooling_index[batch_idx * output_dimension + output_idx] = max_index;
                    }
                }

            });
        }
    }

    public class Layer_Output
    {
        public int Layer_TOP = 0;

        public List<float[]> layerOutputs = new List<float[]>();
        public List<float[]> layerPooling = new List<float[]>();
        public List<int[]> layerMaxPooling_Index = new List<int[]>();
        public List<float[]> layerPoolingSecondary = new List<float[]>();
        public List<int[]> layerMaxPooling_IndexSecondary = new List<int[]>();
        public List<int> LayerDim = new List<int>();
        public Layer_Output(DNN dnn_model)
        {
            foreach (NeuralLink neurallink in dnn_model.neurallinks)
            {
                if (neurallink.Nt == N_Type.Fully_Connected)
                {
                    layerPooling.Add(new float[1]);
                    layerMaxPooling_Index.Add(new int[1]);
                }
                else
                {
                    layerPooling.Add(new float[dnn_model.LFE.MAX_TOKEN_NUM * neurallink.Neural_Out.Number]);
                    layerMaxPooling_Index.Add(new int[neurallink.Neural_Out.Number]);
                }
                layerPoolingSecondary.Add(new float[dnn_model.MaxPoolSentenceNumber * neurallink.Neural_Out.Number]);
                layerMaxPooling_IndexSecondary.Add(new int[neurallink.Neural_Out.Number]);
                
                layerOutputs.Add(new float[neurallink.Neural_Out.Number]);
                
                LayerDim.Add(neurallink.Neural_Out.Number);
            }
            Layer_TOP = layerOutputs.Count - 1;
        }
        ~Layer_Output()
        {
        }
    }

    public class DNN
    {
        List<NeuralLayer> neurallayers = new List<NeuralLayer>();
        public List<NeuralLink> neurallinks = new List<NeuralLink>();

        public Letter_Feature_Extraction LFE = new Letter_Feature_Extraction();
        public int FEATURENUM = 0;
        public int TYPE = 0; // 0 : DSSM; 1 : CDSSM; 2: oldDSSM; 3:oldCDSSM
        public int PoolIdx = 0; // 0 : NonPooling;  1 : Layer 1 Pooling; 2 : Layer 2 Pooling;
        
        int maxPoolSentenceNumber = 30;
        public int MaxPoolSentenceNumber
        {
            get { return maxPoolSentenceNumber; }
        }

        //ModelType only decides the genrc of loading of features
        public DNN(string fileName, ModelType type, string vocabfile, int MaxTokenNum)
        {
            if (type == ModelType.DSSM)
                TYPE = 0;
            else if (type == ModelType.CDSSM)
                TYPE = 1;

            DSSM_Model_Load_NF(fileName);

            if (vocabfile != null && vocabfile != "NULL" && vocabfile != "FEAT")
                LFE.LoadVocab(vocabfile);

            LFE.MAX_TOKEN_NUM = MaxTokenNum;
        }

        /// <summary>
        /// Load DSSM
        /// </summary>
        /// <param name="fileName"></param>
        public void DSSM_Model_Load_NF(string fileName)
        {
            FileStream mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            BinaryReader mreader = new BinaryReader(mstream);

            List<int> layer_info = new List<int>();
            int mlayer_num = mreader.ReadInt32();
            for (int i = 0; i < mlayer_num; i++)
            {
                layer_info.Add(mreader.ReadInt32());
            }

            FEATURENUM = layer_info[0];

            for (int i = 0; i < layer_info.Count; i++)
            {
                NeuralLayer layer = new NeuralLayer(layer_info[i]);
                neurallayers.Add(layer);
            }

            int mlink_num = mreader.ReadInt32();
            for (int i = 0; i < mlink_num; i++)
            {
                int in_num = mreader.ReadInt32();
                int out_num = mreader.ReadInt32();
                float inithidbias = mreader.ReadSingle();
                float initweightsigma = mreader.ReadSingle();
                int mws = mreader.ReadInt32();
                N_Type mnt = (N_Type)mreader.ReadInt32();
                P_Pooling mp = (P_Pooling)mreader.ReadInt32();

                if (mnt == N_Type.Convolution_layer)
                {
                    NeuralLink link = new NeuralLink(neurallayers[i], neurallayers[i + 1], A_Func.Tanh, 1, mnt, mws);
                    neurallinks.Add(link);
                }
                else if(mnt == N_Type.Fully_Connected)
                {
                    NeuralLink link = new NeuralLink(neurallayers[i], neurallayers[i + 1], A_Func.Tanh);
                    neurallinks.Add(link);
                }
            }

            for (int i = 0; i < mlink_num; i++)
            {
                int weight_len = mreader.ReadInt32(); // Write(neurallinks[i].Back_Weight.Length);
                if (weight_len != neurallinks[i].Back_Weight.Length)
                {
                    Console.WriteLine("Loading Model Weight Error!  " + weight_len.ToString() + " " + neurallinks[i].Back_Weight.Length.ToString());
                    Console.ReadLine();
                }
                for (int m = 0; m < weight_len; m++)
                {
                    neurallinks[i].Back_Weight[m] = mreader.ReadSingle();
                }
                int bias_len = mreader.ReadInt32();
                if (bias_len != neurallinks[i].Back_Bias.Length)
                {
                    Console.WriteLine("Loading Model Bias Error!  " + bias_len.ToString() + " " + neurallinks[i].Back_Bias.Length.ToString());
                    Console.ReadLine();
                }
                for (int m = 0; m < bias_len; m++)
                {
                    neurallinks[i].Back_Bias[m] = mreader.ReadSingle();
                }
            }
            mreader.Close();
            mstream.Close();
        }

        public void forward_activate(List<Sample_Input> dataList, Layer_Output output)
        {
            if (dataList.Count == 0)
            {
                return;
            }
            int layerIndex = 0;
            int batchsize = dataList.First().batchsize;

            foreach (NeuralLink neurallink in neurallinks)
            {
                if (layerIndex < PoolIdx)
                {
                    for (int i = 0; i < dataList.Count; ++i)
                    {
                        Sample_Input data = dataList[i];

                        ///first layer.
                        if (layerIndex == 0)
                        {
                            if (neurallink.Nt == N_Type.Fully_Connected)
                            {
                                //????data.Norm_BOW(2);
                                BasicMathlib.Sparse_Matrix_Multiply_INTEX(data.Sample_Idx, data.Fea_Idx, data.Fea_Value, data.elementsize,
                                        neurallink.Back_Weight, output.layerPoolingSecondary[layerIndex], data.batchsize,
                                        neurallink.Neural_Out.Number, neurallink.Neural_In.Number, i * neurallink.Neural_Out.Number);
                            }
                            else if (neurallink.Nt == N_Type.Convolution_layer)
                            {
                                BasicMathlib.Convolution_Sparse_Matrix_Multiply_INTEX(data.Sample_Idx, data.batchsize, data.Seg_Idx, data.Seg_Margin, data.segsize, data.Fea_Idx, data.Fea_Value, data.elementsize,
                                                neurallink.Back_Weight, output.layerPooling[layerIndex], neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);
                                BasicMathlib.Max_Pooling(output.layerPooling[layerIndex], data.Sample_Idx, data.batchsize, output.layerPoolingSecondary[layerIndex], output.layerMaxPooling_Index[layerIndex], neurallink.Neural_Out.Number, i * neurallink.Neural_Out.Number);
                            }
                        }
                        else
                        {
                            BasicMathlib.Matrix_Multiply(output.layerPoolingSecondary[layerIndex - 1], neurallink.Back_Weight, output.layerPoolingSecondary[layerIndex], batchsize, neurallink.Neural_Out.Number, neurallink.Neural_In.Number, i * neurallink.Neural_In.Number, i * neurallink.Neural_Out.Number);
                        }
                    }
                    if (neurallink.Af == A_Func.Tanh)
                    {
                        BasicMathlib.Matrix_Add_Tanh(output.layerPoolingSecondary[layerIndex], neurallink.Back_Bias, batchsize, neurallink.Neural_Out.Number * dataList.Count);
                    }
                    if (layerIndex == PoolIdx - 1)
                    {
                        BasicMathlib.Max_PoolingDense(output.layerPoolingSecondary[layerIndex], batchsize, output.layerOutputs[layerIndex], output.layerMaxPooling_IndexSecondary[layerIndex], dataList.Count, neurallink.Neural_Out.Number);
                    }
                }
                else
                {
                    ///first layer.
                    if (layerIndex == 0)
                    {
                        if (neurallink.Nt == N_Type.Fully_Connected)
                        {
                            BasicMathlib.Sparse_Matrix_Multiply_INTEX(dataList[0].Sample_Idx, dataList[0].Fea_Idx, dataList[0].Fea_Value, dataList[0].elementsize,
                                    neurallink.Back_Weight, output.layerOutputs[layerIndex], dataList[0].batchsize,
                                    neurallink.Neural_Out.Number, neurallink.Neural_In.Number,0);
                        }
                        else if (neurallink.Nt == N_Type.Convolution_layer)
                        {
                            BasicMathlib.Convolution_Sparse_Matrix_Multiply_INTEX(dataList[0].Sample_Idx, dataList[0].batchsize, dataList[0].Seg_Idx, dataList[0].Seg_Margin, dataList[0].segsize, dataList[0].Fea_Idx, dataList[0].Fea_Value, dataList[0].elementsize,
                                            neurallink.Back_Weight, output.layerPooling[layerIndex], neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);

                            BasicMathlib.Max_Pooling(output.layerPooling[layerIndex], dataList[0].Sample_Idx, dataList[0].batchsize, output.layerOutputs[layerIndex], output.layerMaxPooling_Index[layerIndex], neurallink.Neural_Out.Number, 0);
                        }
                    }
                    else
                    {
                        BasicMathlib.Matrix_Multiply(output.layerOutputs[layerIndex - 1], neurallink.Back_Weight, output.layerOutputs[layerIndex], dataList[0].batchsize, neurallink.Neural_Out.Number, neurallink.Neural_In.Number, 0, 0);
                    }
                    if (neurallink.Af == A_Func.Tanh)
                    {
                        BasicMathlib.Matrix_Add_Tanh(output.layerOutputs[layerIndex], neurallink.Back_Bias, dataList[0].batchsize, neurallink.Neural_Out.Number);
                    }
                }
                layerIndex += 1;
            }
        }

        public List<float> Forward(int [] feaidx, float[] feaval)
        {
            Layer_Output output = new Layer_Output(this);
            List<Sample_Input> inputList = new List<Sample_Input>();

            Sample_Input input = new Sample_Input();
            if (TYPE == 0 || TYPE == 2) //DSSM -- bag of word feature
            {
                Dictionary<int, float> fea = new Dictionary<int, float>();

                for (int i = 0; i < feaidx.Length; i++)
                    fea.Add(feaidx[i], feaval[i]);

                input.Load_BOW(fea);
            }
            else if (TYPE == 1 || TYPE == 3) //CDSSM -- seq of word feature
            {
                throw new Exception("cdssm for this kind of feature NOT SUPPORTED yet!");
            }
            inputList.Add(input);

            forward_activate(inputList, output);

            List<float> result = new List<float>();
            for (int i = 0; i < output.layerOutputs[output.Layer_TOP].Length; i++)
                result.Add(output.layerOutputs[output.Layer_TOP][i]);
            return result;
        }

        public List<float> Forward(float [] feavalues)
        {
            Layer_Output output = new Layer_Output(this);
            List<Sample_Input> inputList = new List<Sample_Input>();

            Sample_Input input = new Sample_Input();
            if (TYPE == 0 || TYPE == 2) //DSSM -- bag of word feature
            {
                Dictionary<int, float> fea = new Dictionary<int, float>();

                for (int i = 0; i < feavalues.Length; i++)
                    if (!(feavalues[i] < 0.0000001 && feavalues[i] > -0.0000001)) fea.Add(i, feavalues[i]);

                input.Load_BOW(fea);
            }
            else if (TYPE == 1 || TYPE == 3) //CDSSM -- seq of word feature
            {
                throw new Exception("cdssm for this kind of feature NOT SUPPORTED yet!");
            }
            inputList.Add(input);

            forward_activate(inputList, output);

            List<float> result = new List<float>();
            for (int i = 0; i < output.layerOutputs[output.Layer_TOP].Length; i++)
                result.Add(output.layerOutputs[output.Layer_TOP][i]);
            return result;
        }

        public List<float> Forward(string text, Dictionary<string, Dictionary<int, float>> dic, FeatureList featureList)
        {
            Layer_Output output = new Layer_Output(this);
            List<Sample_Input> inputList = new List<Sample_Input>();

            string[] sentenceList = PoolIdx >= 1 ?
                text.Split(new char[] { '.', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                : new string[] { text };
            if (sentenceList.Length > MaxPoolSentenceNumber)
            {
                string[] sentenceListTmp = new string[MaxPoolSentenceNumber];
                for (int i = 0; i < MaxPoolSentenceNumber - 1; ++i)
                {
                    sentenceListTmp[i] = sentenceList[i];
                }
                sentenceListTmp[MaxPoolSentenceNumber - 1] = string.Join(" ", sentenceList.Skip(MaxPoolSentenceNumber - 1));
                sentenceList = sentenceListTmp;
            }

            foreach (string sentence in sentenceList)
            {
                Sample_Input input = new Sample_Input();


                if (TYPE == 0 || TYPE == 2) //DSSM -- bag of word feature
                {
                    Dictionary<int, float> fea = new Dictionary<int, float>();

                    int pos = 0;
                    if (featureList.l3g == true)
                    {
                        Dictionary<int, float> tmp = LFE.Feature_Extract_BOW(sentence, pos);     //pos
                        fea = fea.Concat(tmp).ToDictionary(k => k.Key, v => v.Value);
                        pos += LFE.vocab_dict.Count;
                    }
                    if (featureList.root == true)
                    {
                        int count = 0;
                        var featStrFeq = TextUtils.String2FeatStrSeq(sentence, 3, 20, FeatureType.root);  // list of root
                        List<Dictionary<int, double>> tmpList = TextUtils.StrFreq2IdFreq(featStrFeq, FeatureType.root, pos, ref count);
                        Dictionary<int, float> tmp = TextUtils.MergeList(tmpList).ToDictionary(k => k.Key, v => (float)(v.Value));
                        fea = fea.Concat(tmp).ToDictionary(k => k.Key, v => v.Value);
                        pos += count;
                    }
                    if (featureList.infl == true)
                    {
                        int count = 0;
                        var featStrFeq = TextUtils.String2FeatStrSeq(sentence, 3, 20, FeatureType.infl);  // list of inflections
                        List<Dictionary<int, double>> tmpList = TextUtils.StrFreq2IdFreq(featStrFeq, FeatureType.infl, pos, ref count);
                        Dictionary<int, float> tmp = TextUtils.MergeList(tmpList).ToDictionary(k => k.Key, v => (float)(v.Value));
                        fea = fea.Concat(tmp).ToDictionary(k => k.Key, v => v.Value);
                        pos += count;
                    }


                    input.Load_BOW(fea);
                }



                //need to updata
                else if (TYPE == 1 || TYPE == 3) //CDSSM -- seq of word feature
                {
                    List<Dictionary<int, float>> feas = new List<Dictionary<int, float>>();

                    //need to update
                    //if (featureType == FeatureType.we)
                    //{
                    //    feas = LFE.Feature_Extractor_SOW(sentence, dic);
                    //}
                    //else
                    {
                        feas = LFE.Feature_Extractor_SOW(sentence);
                    }

                    input.Load_SOW(feas);
                }




                inputList.Add(input);
            }
            
            forward_activate(inputList, output);

            List<float> result = new List<float>();
            for (int i = 0; i < output.layerOutputs[output.Layer_TOP].Length; i++)
                result.Add(output.layerOutputs[output.Layer_TOP][i]);
            return result;
        }
    }

    public class Sample_Input
    {
        public int batchsize;
        public int segsize; // the total length of the full segments.
        public int elementsize;

        public int[] Fea_Idx;
        public float[] Fea_Value;
        public int[] Sample_Idx; // length of the sequence.
        public int[] Seg_Idx; // the index of Segments.
        public int[] Seg_Margin;
	
		
		public void Norm_BOW(int norm)
        {
            if (norm >= 1)
            {
                for (int i = 0; i < batchsize; i++)  //read sample index.
                {
                    int e = Sample_Idx[i];
                    int s = i == 0 ? 0 : Sample_Idx[i - 1];
                    float sum = 0;
                    for (int k = s; k < e; k++)
                    {
                        sum += (float)Math.Pow(Math.Abs(Fea_Value[k]), norm);
                    }
                    sum = (float)Math.Pow(sum, 1.0f / norm) + float.Epsilon;
                    for (int k = s; k < e; k++)
                    {
                        Fea_Value[k] = Fea_Value[k] / sum;
                    }
                }
            }
        }
		
        public void Load_SOW(List<Dictionary<int, float>> feas)
        {
            batchsize = 1;
            segsize = feas.Count;
            elementsize = 0;
            Sample_Idx = new int[1];

            Seg_Idx = new int[segsize];
            Seg_Margin = new int[segsize];
            int i = 0;
            foreach (Dictionary<int, float> mfea in feas)
            {
                elementsize += mfea.Count;
                Seg_Idx[i] = elementsize;
                Seg_Margin[i] = 0;
                i += 1;
            }
            Sample_Idx[0] = i;

            Fea_Idx = new int[elementsize];
            Fea_Value = new float[elementsize];
            i = 0;
            foreach (Dictionary<int, float> mfea in feas)
            {
                foreach (KeyValuePair<int, float> item in mfea)
                {
                    Fea_Idx[i] = item.Key;
                    Fea_Value[i] = item.Value;
                    i = i + 1;
                }
            }
        }

        public void Load_BOW(Dictionary<int,float> fea)
        {
            Sample_Idx = new int[1];
            Fea_Idx = new int[fea.Count];
            Fea_Value = new float[fea.Count];
            batchsize = 1; //read int four-byte.
            elementsize = fea.Count;
            int i = 0;
            foreach (KeyValuePair<int, float> mfea in fea)
            {
                Fea_Idx[i] = mfea.Key;
                Fea_Value[i] = mfea.Value;
                i += 1;
            }
            Sample_Idx[0] = elementsize;
        }
    }

    
}
