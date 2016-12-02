using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;
namespace DSMlib
{
    public class LearningParameters
    {
        public static float learning_rate = 0.1f;
        public static float lr_begin = 0.4f;
        public static float lr_mid = 0.4f;
        public static float lr_latter = 0.02f;
        public static float momentum = 0;
        public static float finalMomentum = 0.02f;
        public static float lrchange = 0.5f;
        public static bool IsrateDown = false;
        public static bool neg_static_sample = false;
        public static float reject_rate = 1.0f;
        public static float down_rate = 1.0f;
        public static float accept_range = 1.0f;
        public static int total_doc_num = 0;
        public static int learn_style = 0; // 0 : mini-batch; 1 : whole-batch;
    }

    public class Program
    {
        static FileStream log_stream = null;//new FileStream(ParameterSetting.Log_FileName + "SEED" + ParameterSetting.RANDOMSEED.ToString(), FileMode.Create, FileAccess.Write);
        static StreamWriter log_writer = null; // new StreamWriter(log_stream);
        
        public static void Print(string mstr)
        {
            Console.WriteLine(mstr);
            if (log_writer != null)
            {
                log_writer.WriteLine(mstr);
                log_writer.Flush();
            }
        }

        public static Stopwatch timer = new Stopwatch();

        public static void Main()
        {
            try
            {
                string logDirecotry = new FileInfo(ParameterSetting.Log_FileName).Directory.FullName;
                if (!Directory.Exists(logDirecotry))
                {
                    Directory.CreateDirectory(logDirecotry);
                }
                log_stream = new FileStream(ParameterSetting.Log_FileName, FileMode.Append, FileAccess.Write);
                log_writer = new StreamWriter(log_stream);

                string modelDirectory = new FileInfo(ParameterSetting.MODEL_PATH).Directory.FullName;
                if (!Directory.Exists(modelDirectory))
                {
                    Directory.CreateDirectory(modelDirectory);
                }

                timer.Reset();
                timer.Start();
                Print("Loading doc Query Stream ....");

                if (ParameterSetting.CuBlasEnable)
                {
                    Cudalib.CUBLAS_Init();
                }
                //Load_Train_PairData(ParameterSetting.QFILE, ParameterSetting.DFILE);

                DNN_Train dnnTrain = null;

                /// 1. loading training dataset.
                dnnTrain = new DSSM_Train();
                dnnTrain.LoadTrainData(new string[] { ParameterSetting.QFILE, ParameterSetting.DFILE });
                if (ParameterSetting.ISVALIDATE)
                {
                    if (!ParameterSetting.VALIDATE_MODEL_ONLY)
                    {
                        dnnTrain.LoadValidateData(new string[] { ParameterSetting.VALIDATE_QFILE, ParameterSetting.VALIDATE_DFILE, ParameterSetting.VALIDATE_QDPAIR });
                    }
                    else
                    {
                        Program.Print("Validation process without stream; model only");
                    }
                }

                /// 2. loading config and start to train.
                dnnTrain.ModelInit_FromConfig();
                dnnTrain.Training();
                dnnTrain.Dispose();
                
                log_writer.Close();
                log_stream.Close();
                if (ParameterSetting.CuBlasEnable)
                {
                    Cudalib.CUBLAS_Destroy();
                }
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine(exc.ToString());
                Environment.Exit(0);
            }
        }
        
        
    }
}
