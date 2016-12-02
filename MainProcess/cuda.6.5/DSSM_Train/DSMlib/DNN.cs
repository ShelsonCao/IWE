using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
namespace DSMlib
{
    public enum A_Func { Linear = 0, Tanh = 1, Rectified = 2 };
    public enum N_Type { Fully_Connected = 0, Convolution_layer = 1 };
    public enum P_Pooling {MAX_Pooling = 0 };
    /// <summary>
    /// Model related parameters
    /// </summary>
    public class NeuralLayer
    {
        public int Number;        
        public NeuralLayer(int num)
        {
            Number = num;            
        }        
    }    
    /// <summary>
    /// Model related parameters
    /// </summary>
    public class NeuralLink : IDisposable
    {
        public NeuralLayer Neural_In;
        public NeuralLayer Neural_Out;

        public CudaPieceFloat weight;
        public CudaPieceFloat bias;

        public IntPtr Weight { get { return weight.CudaPtr; } }
        public IntPtr Bias { get { return bias.CudaPtr; } }

        public float[] Back_Weight { get { return weight.MemPtr; } }
        public float[] Back_Bias { get { return bias.MemPtr; } }

        public A_Func Af;
        public N_Type Nt = N_Type.Fully_Connected;
        public int N_Winsize = 1;

        public P_Pooling pool_type = P_Pooling.MAX_Pooling;

        public float initHidBias = 0;
        public float initWeightSigma = 0.2f;

        unsafe public void CopyOutFromCuda()
        {
            weight.CopyOutFromCuda();
            bias.CopyOutFromCuda();
        }

        unsafe public void CopyIntoCuda()
        {
            weight.CopyIntoCuda();
            bias.CopyIntoCuda();
        }


        public NeuralLink(NeuralLayer layer_in, NeuralLayer layer_out, A_Func af, float hidBias, float weightSigma, N_Type nt, int win_size, bool backupOnly)
        {
            Neural_In = layer_in;
            Neural_Out = layer_out;
            //Neural_In.Number = Neural_In.Number; // *N_Winsize;
            Nt = nt;
            N_Winsize = win_size;

            Af = af;
            initHidBias = hidBias;
            initWeightSigma = weightSigma;

            weight = new CudaPieceFloat(Neural_In.Number * Neural_Out.Number * N_Winsize, true, backupOnly ? false : true);
            bias = new CudaPieceFloat(Neural_Out.Number, true, backupOnly ? false : true);
        }

        ~NeuralLink()
        {
            Dispose();
        }


        public void Dispose()
        {
            weight.Dispose();
            bias.Dispose();            
        }

        public void Init()
        {
            int inputsize = Neural_In.Number * N_Winsize;
            int outputsize = Neural_Out.Number;

            weight.Init((float)(Math.Sqrt(6.0 / (inputsize + outputsize)) * 2), (float)(-Math.Sqrt(6.0 / (inputsize + outputsize))));
            
            //update by Shelson on Jan 27, maybe a bug
            //bias.Init(initHidBias);
            bias.Init((float)(Math.Sqrt(6.0 / (inputsize + outputsize)) * 2), (float)(-Math.Sqrt(6.0 / (inputsize + outputsize))));
        }

        public void Init(float wei_scale, float wei_bias)
        {
            weight.Init(wei_scale, wei_bias);
            bias.Init(initHidBias);
        }

        public void Init(NeuralLink refLink)
        {
            weight.Init(refLink.Back_Weight);
            bias.Init(refLink.Back_Bias);
        }        
    }

    /// <summary>
    /// Model related parameters and network structure
    /// </summary>
    public class DNN
    {
        public List<NeuralLayer> neurallayers = new List<NeuralLayer>();
        public List<NeuralLink> neurallinks = new List<NeuralLink>();

        public DNN(string fileName)
        {
            Model_Load(fileName, true);
        }

        public int OutputLayerSize
        {
            get { return neurallayers.Last().Number; }
        }

        public int ModelParameterNumber
        {
            get
            {
                int NUM = 0;
                for (int i = 0; i < neurallinks.Count; i++)
                {
                    int num = neurallinks[i].Neural_In.Number * neurallinks[i].N_Winsize * neurallinks[i].Neural_Out.Number;
                    if (ParameterSetting.UpdateBias)
                    {
                        num += neurallinks[i].Neural_Out.Number;
                    }
                    NUM += num;
                }
                return NUM;
            }
        }

        public void CopyOutFromCuda()
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].CopyOutFromCuda();
            }
        }

        public void CopyIntoCuda()
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].CopyIntoCuda();
            }
        }
        /// <summary>
        /// For backward-compatible, neurallinks[i].Af = tanh is stored as 0, neurallinks[i].Af = linear is stored as 1, neurallinks[i].Af = rectified is stored as 2 
        /// Do not alter the ordering of those existing A_Func elements.
        /// </summary>
        A_Func[] Int2A_FuncMapping = new A_Func[] {DSMlib.A_Func.Tanh, DSMlib.A_Func.Linear, DSMlib.A_Func.Rectified};
        
        public int A_Func2Int(A_Func af)
        {
            for(int i = 0; i < Int2A_FuncMapping.Length; ++i)
            {
                if(Int2A_FuncMapping[i] == af)
                { 
                    return i; 
                }
            }
            return 0;
        }        
        public A_Func Int2A_Func(int af)
        {
            return Int2A_FuncMapping[af];            
        }
        public void Model_Save(string fileName)
        {
            FileStream mstream = new FileStream(fileName, FileMode.Create, FileAccess.Write);
            BinaryWriter mwriter = new BinaryWriter(mstream);
            mwriter.Write(neurallayers.Count);
            for (int i = 0; i < neurallayers.Count; i++)
            {
                mwriter.Write(neurallayers[i].Number);
            }
            mwriter.Write(neurallinks.Count);
            for (int i = 0; i < neurallinks.Count; i++)
            {
                mwriter.Write(neurallinks[i].Neural_In.Number);
                mwriter.Write(neurallinks[i].Neural_Out.Number);
                mwriter.Write(neurallinks[i].initHidBias);
                mwriter.Write(neurallinks[i].initWeightSigma);
                mwriter.Write(neurallinks[i].N_Winsize);
                //// compose a Int32 integer whose higher 16 bits store activiation function and lower 16 bits store network type
                //// In addition, for backward-compatible, neurallinks[i].Af = tanh is stored as 0, neurallinks[i].Af = linear is stored as 1, neurallinks[i].Af = rectified is stored as 2 
                //// Refer to the Int2A_FuncMapping                
                int afAndNt = ( A_Func2Int(neurallinks[i].Af) << 16) | ((int) neurallinks[i].Nt );
                mwriter.Write(afAndNt);
                mwriter.Write((int)neurallinks[i].pool_type);
            }

            for (int i = 0; i < neurallinks.Count; i++)
            {
                mwriter.Write(neurallinks[i].Back_Weight.Length);
                for (int m = 0; m < neurallinks[i].Back_Weight.Length; m++)
                {
                    mwriter.Write(neurallinks[i].Back_Weight[m]);
                }

                mwriter.Write(neurallinks[i].Neural_Out.Number);
                for (int m = 0; m < neurallinks[i].Neural_Out.Number; m++)
                {
                    mwriter.Write(neurallinks[i].Back_Bias[m]);
                }
            }

            mwriter.Close();
            mstream.Close();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="fileName"></param>
        /// <param name="allocateStructureFromEmpty">True will init DNN structure and allocate new space; False will only load data from file</param>
        public void Model_Load(string fileName, bool allocateStructureFromEmpty)
        {
            FileStream mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            BinaryReader mreader = new BinaryReader(mstream);

            List<int> layer_info = new List<int>();
            int mlayer_num = mreader.ReadInt32();
            for (int i = 0; i < mlayer_num; i++)
            {
                layer_info.Add(mreader.ReadInt32());
            }
            for (int i = 0; i < layer_info.Count; i++)
            {
                if (allocateStructureFromEmpty)
                {
                    NeuralLayer layer = new NeuralLayer(layer_info[i]);
                    neurallayers.Add(layer);
                }
            }

            int mlink_num = mreader.ReadInt32();
            for (int i = 0; i < mlink_num; i++)
            {
                int in_num = mreader.ReadInt32();
                int out_num = mreader.ReadInt32();
                float inithidbias = mreader.ReadSingle();
                float initweightsigma = mreader.ReadSingle();

                NeuralLink link = null;
                if (ParameterSetting.LoadModelOldFormat)
                {
                    if (allocateStructureFromEmpty)
                    {
                        // for back-compatibility only. The old model format donot have those three fields
                        link = new NeuralLink(neurallayers[i], neurallayers[i + 1], A_Func.Tanh, 0, initweightsigma, N_Type.Fully_Connected, 1, false);
                    }
                }
                else
                {
                    // this is the eventually favorable loading format
                    int mws = mreader.ReadInt32();
                    //// decompose a Int32 integer, whose higher 16 bits store activiation function and lower 16 bits store network type
                    //// In addition, for backward-compatible, neurallinks[i].Af = tanh is stored as 0, neurallinks[i].Af = linear is stored as 1, neurallinks[i].Af = rectified is stored as 2 
                    //// Refer to the Int2A_FuncMapping                
                    int afAndNt = mreader.ReadInt32();
                    A_Func aF = Int2A_Func(afAndNt >> 16);
                    N_Type mnt = (N_Type) (afAndNt & ((1<<16)-1));
                    P_Pooling mp = (P_Pooling)mreader.ReadInt32();
                    if (allocateStructureFromEmpty)
                    {
                        link = new NeuralLink(neurallayers[i], neurallayers[i + 1], aF, 0, initweightsigma, mnt, mws, false);
                    }
                }
                if (allocateStructureFromEmpty)
                {
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
            CopyIntoCuda();
        }

        public void Fill_Layer_One(string fileName)
        {
            FileStream mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            BinaryReader mreader = new BinaryReader(mstream);

            List<int> layer_info = new List<int>();
            int mlayer_num = mreader.ReadInt32();
            for (int i = 0; i < mlayer_num; i++)
            {
                layer_info.Add(mreader.ReadInt32());
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

            for (int i = 1; i < neurallinks.Count; i++)
            {
                for (int m = 0; m < neurallinks[i].Back_Bias.Length; m++)
                {
                    neurallinks[i].Back_Bias[m] = 0;
                }
                int wei_num = neurallinks[i].Back_Weight.Length;
                for (int m = 0; m < neurallinks[i].Neural_Out.Number; m++)
                {
                    neurallinks[i].Back_Weight[(m * neurallinks[i].Neural_Out.Number) % wei_num + m] = 1.0f;
                }
            }

            CopyIntoCuda();
        }

        public string DNN_Descr()
        {
            string result = "";
            for (int i = 0; i < neurallayers.Count; i++)
            {
                result += "Neural Layer " + i.ToString() + ": " + neurallayers[i].Number.ToString() + "\n";
            }

            for (int i = 0; i < neurallinks.Count; i++)
            {
                result += "layer " + i.ToString() + " to layer " + (i + 1).ToString() + ":" +
                        " AF Type : " + neurallinks[i].Af.ToString() + ";" +
                        " hid bias : " + neurallinks[i].initHidBias.ToString() + ";" +
                        " weight sigma : " + neurallinks[i].initWeightSigma.ToString() + ";" +
                        " Neural Type : " + neurallinks[i].Nt.ToString() + ";" +
                        " Window Size : " + neurallinks[i].N_Winsize.ToString() + ";" + "\n";
            }
            return result;
        }

        public DNN(int featureSize, int[] layerDim, int[] activation, float[] sigma, int[] arch, int[] wind, bool backupOnly)
        {
            NeuralLayer inputlayer = new NeuralLayer(featureSize);
            neurallayers.Add(inputlayer);
            for (int i = 0; i < layerDim.Length; i++)
            {
                NeuralLayer layer = new NeuralLayer(layerDim[i]);
                neurallayers.Add(layer);
            }

            for (int i = 0; i < layerDim.Length; i++)
            {
                NeuralLink link = new NeuralLink(neurallayers[i], neurallayers[i + 1], (A_Func)activation[i], 0, sigma[i],
                (N_Type)arch[i], wind[i], backupOnly);
                neurallinks.Add(link);
            }
        }
        
        public void Init()
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].Init();
            }
        }

        public void Init(DNN model)
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].Init(model.neurallinks[i]);
            }
        }

        public void Init(float wei_scale, float wei_bias)
        {
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].Init(wei_scale, wei_bias);
            }
        }

        /// <summary>
        /// Before call this stuff, you must call CopyOutFromCuda()
        /// The returns is only used for backup purpose. So its does not allocate any GPU memory.
        /// </summary>
        /// <returns></returns>
        public DNN CreateBackupClone()
        {
            DNN backupClone = new DNN(
                this.neurallayers[0].Number,
                this.neurallinks.Select(o => o.Neural_Out.Number).ToArray(),
                this.neurallinks.Select(o => (int)o.Af).ToArray(),
                this.neurallinks.Select(o => o.initWeightSigma).ToArray(),
                this.neurallinks.Select(o => (int)o.Nt).ToArray(),
                this.neurallinks.Select(o => o.N_Winsize).ToArray(),
                true);
            backupClone.Init(this);
            return backupClone;
        }
    }

    /// <summary>
    /// A particular run related data, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class NeuralLayerData : IDisposable
    {
        NeuralLayer LayerModel;
        public int Number { get { return LayerModel.Number; } }
        /// <summary>        
        /// </summary>
        /// <param name="num"></param>
        /// <param name="isValueNeeded">To save GPU memory, when no errors are needed, we should not allocate error piece. This usually happens on the input layer</param>
        public NeuralLayerData(NeuralLayer layerModel, bool isValueNeeded)
        {
            LayerModel = layerModel;
            if (isValueNeeded)
            {
                output = new CudaPieceFloat(ParameterSetting.BATCH_SIZE * Number, true, true);
                errorDeriv = new CudaPieceFloat(ParameterSetting.BATCH_SIZE * Number, true, true);
            }
        }
        /// <summary>
        /// The output of the layer, i.e., the actual activitation values
        /// </summary>
        CudaPieceFloat output = null;

        public CudaPieceFloat Output
        {
            get { return output; }
        }
        /// <summary>
        /// The error of the layer, back-propagated from the top loss function
        /// </summary>
        CudaPieceFloat errorDeriv = null;

        public CudaPieceFloat ErrorDeriv
        {
            get { return errorDeriv; }
        }

        ~NeuralLayerData()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (output != null)
            {
                output.Dispose();
            }
            if (errorDeriv != null)
            {
                errorDeriv.Dispose();
            }
        }
    }
    /// <summary>
    /// A particular run related data, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class NeuralLinkData :IDisposable
    {
        NeuralLink neuralLinkModel;

        public NeuralLink NeuralLinkModel
        {
            get { return neuralLinkModel; }
        }
        
        /// <summary>
        /// Used if convolutional
        /// </summary>
        CudaPieceFloat layerPoolingOutput = null;

        public CudaPieceFloat LayerPoolingOutput
        {
            get { return layerPoolingOutput; }
        }
        /// <summary>
        /// Used if convolutional and maxpooling
        /// </summary>
        CudaPieceInt layerMaxPooling_Index = null;

        public CudaPieceInt LayerMaxPooling_Index
        {
            get { return layerMaxPooling_Index; }
        }

        CudaPieceFloat weightDeriv = null;

        public CudaPieceFloat WeightDeriv
        {
            get { return weightDeriv; }
        }
        CudaPieceFloat biasDeriv = null;

        public CudaPieceFloat BiasDeriv
        {
            get { return biasDeriv; }
        }

        /// <summary>
        /// Wei_Update = momentum * Wei_Update + learn_rate * grad.
        /// Weight = Weight + Wei_Update
        /// </summary>
        CudaPieceFloat weightUpdate = null;

        public CudaPieceFloat WeightUpdate
        {
            get { return weightUpdate; }
        }
        CudaPieceFloat biasUpdate = null;

        public CudaPieceFloat BiasUpdate
        {
            get { return biasUpdate; }
        }

        public NeuralLinkData(NeuralLink neuralLink)
        {
            neuralLinkModel = neuralLink;
            
            if (neuralLinkModel.Nt == N_Type.Convolution_layer)
            {
                layerPoolingOutput = new CudaPieceFloat(PairInputStream.MAXSEGMENT_BATCH * neuralLinkModel.Neural_Out.Number, false, true);

                layerMaxPooling_Index = new CudaPieceInt(ParameterSetting.BATCH_SIZE * neuralLinkModel.Neural_Out.Number, false, true);
            }

            weightDeriv = new CudaPieceFloat(neuralLinkModel.Neural_In.Number * neuralLinkModel.Neural_Out.Number * neuralLinkModel.N_Winsize, false, true);
            biasDeriv = new CudaPieceFloat(neuralLinkModel.Neural_Out.Number, false, true);

            weightUpdate = new CudaPieceFloat(neuralLinkModel.Neural_In.Number * neuralLinkModel.Neural_Out.Number * neuralLinkModel.N_Winsize, false, true);
            biasUpdate = new CudaPieceFloat(neuralLinkModel.Neural_Out.Number, false, true);
        }

        ~NeuralLinkData()
        {
            Dispose();
        }


        public void Dispose()
        {
            if (layerPoolingOutput != null)
            {
                layerPoolingOutput.Dispose();
            }
            if (layerMaxPooling_Index != null)
            {
                layerMaxPooling_Index.Dispose();
            }
            if (weightDeriv != null)
            {
                weightDeriv.Dispose();
            }
            if (biasDeriv != null)
            {
                biasDeriv.Dispose();
            }
            if (weightUpdate != null)
            {
                weightUpdate.Dispose();
            }
            if (biasUpdate != null)
            {
                biasUpdate.Dispose();
            }
        }        

        public void ZeroDeriv()
        {
            weightDeriv.Zero();
            biasDeriv.Zero();
        }
    }

    /// <summary>
    /// A particular run related data, including feedforward outputs and backward propagation derivatives
    /// </summary>
    public class DNNRun
    {
        public DNN DnnModel = null;
        public List<NeuralLayerData> neurallayers = new List<NeuralLayerData>();
        public List<NeuralLinkData> neurallinks = new List<NeuralLinkData>();

        public DNNRun(DNN model)
        {
            DnnModel = model;
            for(int i = 0; i < DnnModel.neurallayers.Count; ++i)
            {
                neurallayers.Add(new NeuralLayerData(DnnModel.neurallayers[i], i != 0));
            }

            for(int i = 0; i < DnnModel.neurallinks.Count; ++i)
            {
                neurallinks.Add(new NeuralLinkData(DnnModel.neurallinks[i]));
            }
        }

        public int OutputLayerSize
        {
            get { return neurallayers.Last().Number; }
        }

        /// <summary>
        /// given batch of input data. calculate the output.
        /// </summary>
        /// <param name="data"></param>
        //unsafe public void forward_activate( BatchSample_Input data, List<Amplib.AMPArrayInternal> layerOutputs)
        unsafe public void forward_activate(BatchSample_Input data)
        {
            int layerIndex = 0;
            foreach (NeuralLinkData neurallinkData in neurallinks)
            {
                NeuralLink neurallink = neurallinkData.NeuralLinkModel;
                ///first layer.
                if (layerIndex == 0)
                {
                    if (neurallink.Nt == N_Type.Fully_Connected)
                    {
                        MathOperatorManager.GlobalInstance.SEQ_Sparse_Matrix_Multiply_INTEX(data, neurallink.weight, neurallayers[layerIndex + 1].Output,
                                        neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);
                    }
                    else if (neurallink.Nt == N_Type.Convolution_layer)
                    {
                        MathOperatorManager.GlobalInstance.Convolution_Sparse_Matrix_Multiply_INTEX(data, neurallink.weight, neurallinkData.LayerPoolingOutput, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, neurallink.N_Winsize);

                        MathOperatorManager.GlobalInstance.Max_Pooling(neurallinkData.LayerPoolingOutput, data, neurallayers[layerIndex + 1].Output, neurallinkData.LayerMaxPooling_Index, neurallink.Neural_Out.Number);
                    }
                }
                else
                {
                    MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[layerIndex].Output, neurallink.weight, neurallayers[layerIndex + 1].Output, data.batchsize, neurallink.Neural_In.Number, neurallink.Neural_Out.Number, 0);
                }
                if (neurallink.Af == A_Func.Tanh)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Add_Tanh(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                }
                else if (neurallink.Af == A_Func.Linear)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Add_Vector(neurallayers[layerIndex + 1].Output, neurallink.bias, data.batchsize, neurallink.Neural_Out.Number);
                }
                layerIndex += 1;
            }
        }

        /// <summary>
        /// BackProp the error derivative on the output of each layer.
        /// The output layer's errorDeriv must be previuosly setup.
        /// </summary>
        unsafe public void backward_calculate_layerout_deriv(int batchsize)
        {
            for (int i = neurallinks.Count - 1; i > 0; i--)
            {
                MathOperatorManager.GlobalInstance.Matrix_Multipy(neurallayers[i + 1].ErrorDeriv, neurallinks[i].NeuralLinkModel.weight, neurallayers[i].ErrorDeriv, batchsize,
                        neurallinks[i].NeuralLinkModel.Neural_Out.Number, neurallinks[i].NeuralLinkModel.Neural_In.Number, 1);
                if (neurallinks[i - 1].NeuralLinkModel.Af == A_Func.Tanh)
                {
                    MathOperatorManager.GlobalInstance.Deriv_Tanh(neurallayers[i].ErrorDeriv, neurallayers[i].Output, batchsize, neurallinks[i].NeuralLinkModel.Neural_In.Number);
                }
            }
        }

        unsafe public void backward_calculate_weight_deriv(BatchSample_Input input_batch) //, int alpha_index) // float[] alpha)
        {
            int batchsize = input_batch.batchsize;
            for (int i = 0; i < neurallinks.Count; i++)
            {
                neurallinks[i].ZeroDeriv();

                if (ParameterSetting.UpdateBias)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Aggragate(neurallayers[i + 1].ErrorDeriv, neurallinks[i].BiasDeriv, input_batch.batchsize, neurallinks[i].NeuralLinkModel.Neural_Out.Number);
                }

                if (i == 0)
                {
                    if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Fully_Connected)
                    {
                        MathOperatorManager.GlobalInstance.SEQ_Sparse_Matrix_Transpose_Multiply_INTEX(input_batch, neurallinks[i].WeightDeriv, neurallayers[i + 1].ErrorDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number, neurallinks[i].NeuralLinkModel.Neural_Out.Number, neurallinks[i].NeuralLinkModel.N_Winsize);
                    }
                    else if (neurallinks[i].NeuralLinkModel.Nt == N_Type.Convolution_layer)
                    {
                        MathOperatorManager.GlobalInstance.Convolution_Sparse_Matrix_Product_INTEX(neurallayers[i + 1].ErrorDeriv, neurallinks[i].LayerMaxPooling_Index, input_batch, neurallinks[i].NeuralLinkModel.N_Winsize,
                                     batchsize, neurallayers[i + 1].Number, neurallinks[i].WeightDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number);
                    }
                }
                else
                {
                    MathOperatorManager.GlobalInstance.Matrix_Product(neurallayers[i].Output, neurallayers[i + 1].ErrorDeriv, neurallinks[i].WeightDeriv,
                        batchsize, neurallayers[i].Number, neurallayers[i + 1].Number);
                }
            }
        }
                
        /// <summary>
        /// the error deriv at top output layer must have been set up before call this method.
        /// This process only do backprop computations. It does not update model weights at all.
        /// Need to call update_weight afterwards to update models.
        /// </summary>
        /// <param name="input_batch"></param>
        /// <param name="momentum"></param>
        /// <param name="learning_rate"></param>
        public void backward_propagate_deriv(BatchSample_Input input_batch)
        {
            // step 1, compute the derivatives for the output values of each layer
            backward_calculate_layerout_deriv(input_batch.batchsize);
            // step 2, compute the derivatives for the connections of each neural link layer
            backward_calculate_weight_deriv(input_batch);            
        }

        /// <summary>
        /// Must call backward_propagate(), or two steps one by one, before calling this method.
        /// </summary>
        unsafe public void update_weight(float momentum, float learning_rate)
        {
            /// step 1, compute the weight updates, taking momentum and learning rates into consideration
            /// Wei_Update = momentum * Wei_Update + learn_rate * grad.
            for (int i = 0; i < neurallinks.Count; i++)
            {
                // add the momentum
                MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].WeightUpdate, neurallinks[i].NeuralLinkModel.Neural_In.Number * neurallinks[i].NeuralLinkModel.N_Winsize, neurallinks[i].NeuralLinkModel.Neural_Out.Number, momentum);

                // dnn_neurallinks[i].Weight
                MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].WeightUpdate, neurallinks[i].WeightDeriv, neurallinks[i].NeuralLinkModel.Neural_In.Number * neurallinks[i].NeuralLinkModel.N_Winsize, neurallinks[i].NeuralLinkModel.Neural_Out.Number, learning_rate);
                
                //dnn_model.neurallinks[i].Bias
                if (ParameterSetting.UpdateBias)
                {
                    // add the momentum
                    MathOperatorManager.GlobalInstance.Scale_Matrix(neurallinks[i].BiasUpdate, 1, neurallinks[i].NeuralLinkModel.Neural_Out.Number, momentum);

                    // dnn_neurallinks[i].Weight
                    MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].BiasUpdate, neurallinks[i].BiasDeriv, 1, neurallinks[i].NeuralLinkModel.Neural_Out.Number, learning_rate);
                }
            }

            // step 2, update the model: Weight = Weight += Wei_Update
            for (int i = 0; i < neurallinks.Count; i++)
            {
                // dnn_model.neurallinks[i].Weight
                MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].NeuralLinkModel.weight, neurallinks[i].WeightUpdate, neurallinks[i].NeuralLinkModel.Neural_In.Number * neurallinks[i].NeuralLinkModel.N_Winsize, neurallinks[i].NeuralLinkModel.Neural_Out.Number, 1.0f);

                if (ParameterSetting.UpdateBias)
                {
                    MathOperatorManager.GlobalInstance.Matrix_Add(neurallinks[i].NeuralLinkModel.bias, neurallinks[i].BiasUpdate, 1, neurallinks[i].NeuralLinkModel.Neural_Out.Number, 1.0f);
                }
                //dnn_model.neurallinks[i].Bias
                //Cudalib.Matrix_Add(dnn_model_query.neurallinks[i].Bias, Wei_Update_query.Layer_Bias[i], 1, dnn_model_query.neurallinks[i].Neural_Out.Number, 1.0f);
            }
        }
    }
}
