using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WordHash;
using ComputeLogPD;
using DSMlib;
using sent2vec;
using jlib;
using Util;
using System.Configuration;
using System.IO;


namespace MainProcess
{
    class Program
    {
        public void GetQryTrgFea(string trainPairTokzPath, string feaFilePath, string trainSrcSeqBinPath,
    string trainTgtSeqBinPath, int srcShortTxtWinSize, int tgtShortTxtWinSize, int batchSize, FeatureList featureList)
        {
            WordHash.Program.Pair2SeqFeaBin(trainPairTokzPath, feaFilePath, srcShortTxtWinSize, 0, trainSrcSeqBinPath, batchSize, featureList);
            WordHash.Program.Pair2SeqFeaBin(trainPairTokzPath, feaFilePath, tgtShortTxtWinSize, 1, trainTgtSeqBinPath, batchSize, featureList);
        }

        public void CptLogPD(string trainPairTokzPath, string trainLogPDPath)
        {
            string argTerms = "/i " + trainPairTokzPath + " /o " + trainLogPDPath + " /C 1 /S 0.75";
            string[] args = argTerms.Split(' ');

            ComputeLogPD.Program.Main(args);
        }

        public void RunCDSSM()
        {
            DSMlib.Program.Main();
        }

        public void Embed(string docDoneFile, string feaFilePath, string dicPath, string embeddings,
            int tgtShortTxtWinSize, int dim, string modelType, FeatureList featureList)
        {
            ModelType mt = ModelType.CDSSM;
            if (modelType.Equals("DSSM"))
            {
                mt = ModelType.DSSM;
            }

            try
            {
                sent2vec.Program.Embedding(docDoneFile, feaFilePath, mt, tgtShortTxtWinSize, dicPath, embeddings, dim, featureList);
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine(exc.ToString());
                Environment.Exit(0);
            }
        }

        private static void RunPipeline()
        {
            Program p = new Program();
            PairGenerator gen = new PairGenerator(ParameterSetting.CORPUS, ParameterSetting.DIC, ParameterSetting.l3gPath);

            if (!File.Exists(ParameterSetting.trainPairTokz))
            {
                Console.WriteLine("Generate Query and Target Files...");
                gen.GenQryTrgFile(ParameterSetting.CORPUS, ParameterSetting.trainPairTokz);
            }

            if ((!File.Exists(ParameterSetting.QFILE)) && (!File.Exists(ParameterSetting.DFILE)))
            {
                Console.WriteLine("Get Query and Target Features...");
                p.GetQryTrgFea(ParameterSetting.trainPairTokz, ParameterSetting.l3gPath, ParameterSetting.QFILE, ParameterSetting.DFILE, 
                    ParameterSetting.srcShortTxtWinSize, ParameterSetting.tgtShortTxtWinSize, ParameterSetting.BATCH_SIZE, ParameterSetting.featureList);
            }

            if (!File.Exists(ParameterSetting.NCE_PROB_FILE))
            {
                Console.WriteLine("Compute Log Probability...");
                if (File.Exists(ParameterSetting.trainPairTokzNew))
                {
                    p.CptLogPD(ParameterSetting.trainPairTokzNew, ParameterSetting.NCE_PROB_FILE);
                    File.Delete(ParameterSetting.trainPairTokzNew);
                    File.Delete(ParameterSetting.trainPairTokz);
                }
            }

            Console.WriteLine("Run CDSSM Model...");

            p.RunCDSSM();

            int dim = ParameterSetting.TARGET_LAYER_DIM[ParameterSetting.TARGET_LAYER_DIM.Length - 1];
            p.Embed(ParameterSetting.docDoneFile, ParameterSetting.l3gPath, ParameterSetting.DIC, ParameterSetting.EMB_FILE, 
                ParameterSetting.tgtShortTxtWinSize, dim, ParameterSetting.tgtModelType, ParameterSetting.featureList);
        }


        static void Main(string[] args)
        {
            if (args.Length <= 0)
            {
                Console.WriteLine("Please specify config file:");
                return;
            }
            ParameterSetting.LoadArgs(args[0]);

            RunPipeline();
        }
    }
}
