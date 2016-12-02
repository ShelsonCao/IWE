using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

using System.Diagnostics;
using System.Runtime.InteropServices;

namespace DSMlib
{
    public enum Evaluation_Type
    {
        PairScore,
        MultiRegressionScore,
        ClassficationScore,
        PairRegressioinScore
    }

    public class SequenceInputStream:IDisposable
    {
        public BatchSample_Input Data = null;

        public int total_Batch_Size = 0;
        public int Feature_Size = 0;
        public int MAXELEMENTS_PERBATCH = 0;
        public int MAXSEQUENCE_PERBATCH = 0;
        public int BATCH_NUM = 0;
        public int BATCH_INDEX = 0;
        public int LAST_INCOMPLETE_BATCH_SIZE = 0;
        FileStream mstream = null;
        BinaryReader mreader = null;

        ~SequenceInputStream()
        {
            Dispose();
        }

        public void CloseStream()
        {
            if(mstream != null)
            {
                mreader.Close();
                mstream.Close();
                mreader = null;
                mstream = null;
            }
            if(Data != null)
            {
                Data.Dispose();
                Data = null;
            }
        }

        public void get_dimension(string fileName)
        {
            if (ParameterSetting.LoadInputBackwardCompatibleMode == "BOW")
            {
                // code for back-compatibility to the previous BOW bin format
                mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                mreader = new BinaryReader(mstream);
                mstream.Seek(-3 * sizeof(Int32), SeekOrigin.End);

                Feature_Size = mreader.ReadInt32(); //// binary feature file stores feature dimension
                total_Batch_Size = mreader.ReadInt32();                
                MAXELEMENTS_PERBATCH = mreader.ReadInt32();
                MAXSEQUENCE_PERBATCH = ParameterSetting.BATCH_SIZE;                
            }
            else if (ParameterSetting.LoadInputBackwardCompatibleMode == "SEQ")
            {
                // code for back-compatibility to the previous SEQ bin format, with unnecessary batch_size and feature_dim
                mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                mreader = new BinaryReader(mstream);
                mstream.Seek(-4 * sizeof(Int32), SeekOrigin.End);

                Feature_Size = mreader.ReadInt32(); //// binary feature file stores feature dimension           
                total_Batch_Size = mreader.ReadInt32();
                MAXSEQUENCE_PERBATCH = mreader.ReadInt32();
                MAXELEMENTS_PERBATCH = mreader.ReadInt32();
            }
            else
            {
                mstream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
                mreader = new BinaryReader(mstream);
                mstream.Seek(-5 * sizeof(Int32), SeekOrigin.End);

                Feature_Size = mreader.ReadInt32(); //// binary feature file stores feature dimension           
                total_Batch_Size = mreader.ReadInt32();
                MAXSEQUENCE_PERBATCH = mreader.ReadInt32();
                MAXELEMENTS_PERBATCH = mreader.ReadInt32();
                int batch_size = mreader.ReadInt32();
                if (batch_size != ParameterSetting.BATCH_SIZE)
                {
                    throw new Exception(string.Format(
                        "Batch_Size does not match between configuration and input data!\n\tFrom config: {0}.\n\tFrom data ({1}): {2}"
                        , ParameterSetting.BATCH_SIZE, fileName, batch_size)
                        );
                }
            }
            Data = new BatchSample_Input(ParameterSetting.BATCH_SIZE, MAXSEQUENCE_PERBATCH, MAXELEMENTS_PERBATCH);
            
            BATCH_NUM = (total_Batch_Size + ParameterSetting.BATCH_SIZE - 1) / ParameterSetting.BATCH_SIZE;
            LAST_INCOMPLETE_BATCH_SIZE = total_Batch_Size % ParameterSetting.BATCH_SIZE;
            BATCH_INDEX = 0;
        }
        
        public void Init()
        {
            BATCH_INDEX = 0;
            mstream.Seek(0, SeekOrigin.Begin);
        }

        void LoadDataBatch(int allowedFeatureDimension)
        {
            int expectedBatchSize = ParameterSetting.BATCH_SIZE;
            if(BATCH_INDEX == BATCH_NUM - 1 && LAST_INCOMPLETE_BATCH_SIZE != 0)
            {
                // only when the last batch is less than BATCH_SIZE, we will need some care
                expectedBatchSize = LAST_INCOMPLETE_BATCH_SIZE;
            }

            if(Feature_Size <= allowedFeatureDimension)
            {
                //// if the feature_size of the entire stream is not bigger than the allowed feature dimension,
                //// we just load the batch without the need of reading perbatch feature size or filtering OOV features
                Data.Load(mreader, expectedBatchSize, false);
            }
            else
            {
                //// if the feature_size of the entire stream is *bigger than* the allowed feature dimension,
                //// we need to read the feature size of the current batch, 
                //// and filter OOV feature if the feature size of the current batch is greater than the allowed feature dimension
                int featureDimInThisBatch = Data.Load(mreader, expectedBatchSize, true);
                if (featureDimInThisBatch > allowedFeatureDimension)
                {
                    Data.FilterOOVFeature(allowedFeatureDimension);
                }
            }
        }

        public bool Fill(int allowedFeatureDimension)
        {
            if (BATCH_INDEX == BATCH_NUM)
            {
                return false;
            }
            LoadDataBatch(allowedFeatureDimension);            
            BATCH_INDEX++;
            return true;
        }

        public void Dispose()
        {            
            CloseStream();
        }
    }

    public class EvaluationSet
    {
        public static EvaluationSet Create(Evaluation_Type type)
        {
            EvaluationSet eval = null;
            switch (type)
            {
                case Evaluation_Type.PairScore:
                    eval = new PairScoreEvaluationSet();
                    break;
                case Evaluation_Type.MultiRegressionScore:
                    eval = new MultiRegressionEvaluationSet();
                    break;
                case Evaluation_Type.ClassficationScore:
                    eval = new ClassificationEvaluationSet();
                    break;
                case Evaluation_Type.PairRegressioinScore:
                    eval = new RegressionEvaluationSet();
                    break;
            }
            return eval;
        }

        public virtual void Loading_LabelInfo(string[] files)
        { }

        public virtual void Ouput_Batch(float[] score,  float[] groundTrue, int[] args)
        { }

        public virtual void Init()
        { }

        public virtual void Save(string scoreFile)
        { }

        static void CallExternalMetricEXE(string executiveFile, string arguments, string metricEvalResultFile)
        {
            if (File.Exists(metricEvalResultFile))
            {
                // remove previous result
                File.Delete(metricEvalResultFile);
            }
            using (Process callProcess = new Process()
            {
                StartInfo = new ProcessStartInfo()
                {
                    FileName = executiveFile,
                    Arguments = arguments,
                    CreateNoWindow = false,
                    UseShellExecute = true,
                }
            })
            {
                callProcess.Start();
                callProcess.WaitForExit();
            }
        }

        static float ReadExternalObjectiveMetric(string metricEvalResultFile, out List<string> validationFileLines)
        {
            // the first line should be a float, specifying the metric score
            if (!File.Exists(metricEvalResultFile))
            {
                throw new Exception(string.Format("Missing objective metric result file {0}, check your validation evaluation process!", metricEvalResultFile));
            }
            validationFileLines = new List<string>();
            
            StreamReader sr = new StreamReader(metricEvalResultFile);
            string line = sr.ReadLine();    // read the first line only
            float objectiveMetric = 0;
            if (!float.TryParse(line, out objectiveMetric))
                throw new Exception(string.Format("Cannot read objective metric from the result file {0}, check your validation evaluation process!", metricEvalResultFile));
            validationFileLines.Add(line);
            while ((line = sr.ReadLine()) != null)
            {
                validationFileLines.Add(line);
            }
            sr.Close();

            return objectiveMetric;
        }

        public float Evaluation(out List<string> validationFileLines)
        {
            string pairScoreFile = Path.GetRandomFileName();
            string objectiveMetricFile = pairScoreFile + ".metric";
            Program.Print("Saving validation prediction scores ... ");
            
            Save(pairScoreFile);
            //PairValidStream.SavePairPredictionScore(pairScoreFile);

            Program.Print("Calling external validation process ... ");
            CallExternalMetricEXE(ParameterSetting.VALIDATE_PROCESS, string.Format("{0} {1}", pairScoreFile, objectiveMetricFile), objectiveMetricFile);

            Program.Print("Reading validation objective metric  ... ");
            validationFileLines = null;
            float result = ReadExternalObjectiveMetric(objectiveMetricFile, out validationFileLines);

            if (File.Exists(pairScoreFile))
            {
                File.Delete(pairScoreFile);
            }

            if (File.Exists(objectiveMetricFile))
            {
                File.Delete(objectiveMetricFile);
            }

            return result;
        }

        public static float EvaluationModelOnly(string srcModelPath, string tgtModelPath, out List<string> validationFileLines)
        {
            string validationResultFile = Path.GetRandomFileName();

            Program.Print("Calling external validation process ... ");
            PairScoreEvaluationSet.CallExternalMetricEXE(ParameterSetting.VALIDATE_PROCESS, string.Format("{0} {1} {2}", srcModelPath, tgtModelPath, validationResultFile), validationResultFile);

            Program.Print("Reading validation objective metric  ... ");
            
            float result = PairScoreEvaluationSet.ReadExternalObjectiveMetric(validationResultFile, out validationFileLines);

            if (File.Exists(validationResultFile))
            {
                File.Delete(validationResultFile);
            }

            return result;
        }
    }

    public class PairScoreEvaluationSet : EvaluationSet
    {
        public List<string> PairInfo_Details = new List<string>();
        public string PairInfo_Header = string.Empty;
        public List<float> Pair_Score = new List<float>();

        public override void Loading_LabelInfo(string[] files)
        {
            FileStream mstream = new FileStream(files[0], FileMode.Open, FileAccess.Read);
            StreamReader mreader = new StreamReader(mstream);

            string mline = mreader.ReadLine();
            int Line_Idx = 0;
            if (!mline.Contains("m:"))
            {
                PairInfo_Details.Add(mline);
                Line_Idx += 1;
            }
            else
            {
                PairInfo_Header = mline;
            }
            while (!mreader.EndOfStream)
            {
                mline = mreader.ReadLine();
                PairInfo_Details.Add(mline);
                Line_Idx += 1;
            }
            mreader.Close();
            mstream.Close();
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            mtwriter.WriteLine(PairInfo_Header + "\tDSSM_Score"); // header
            for (int i = 0; i < Pair_Score.Count; i++)
            {
                float v = Pair_Score[i];
                mtwriter.WriteLine(PairInfo_Details[i] + "\t" + v.ToString());
            }
            mtwriter.Close();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            for (int i = 0; i < args[0]; i++)
            {
                Pair_Score.Add(score[i]);
            }
        }

        public override void Init()
        {
            Pair_Score.Clear();
        }
    }

    public class MultiRegressionEvaluationSet : EvaluationSet
    {
        public List<string> PairInfo_Details = new List<string>();
        public List<float> Pair_Score = new List<float>();
        public int Dimension = 0;
        public override void Init()
        {
 	        PairInfo_Details.Clear();
            Pair_Score.Clear();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            Dimension = args[1];
            for (int i = 0; i < args[0]; i++)
            {
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < args[1]; k++)
                {
                    Pair_Score.Add(score[i * args[1] + k]);
                    sb.Append(groundTrue[i * args[1] + k].ToString() + ",");
                }
                PairInfo_Details.Add(sb.ToString());
            }
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            for (int i = 0; i < Pair_Score.Count / Dimension; i++)
            {
                StringBuilder sb = new StringBuilder();
                for (int k = 0; k < Dimension; k++)
                {
                    sb.Append(Pair_Score[i * Dimension + k].ToString() + ",");
                }
                mtwriter.WriteLine(PairInfo_Details[i].ToString() + "\t" + sb.ToString());
            }
            mtwriter.Close();
        }
    }

    public class RegressionEvaluationSet : EvaluationSet
    {
        public List<float> PairInfo_Details = new List<float>();
        public List<float> Pair_Scores = new List<float>();

        public override void Loading_LabelInfo(string[] files)
        {
            FileStream mstream = new FileStream(files[0], FileMode.Open, FileAccess.Read);
            StreamReader mreader = new StreamReader(mstream);

            int Line_Idx = 0;
            while (!mreader.EndOfStream)
            {
                string mline = mreader.ReadLine();
                float label = float.Parse(mline);
                PairInfo_Details.Add(label);
                Line_Idx += 1;
            }
            mreader.Close();
            mstream.Close();
        }

        public override void Init()
        {
            Pair_Scores.Clear();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            for (int i = 0; i < args[0]; i++)
            {
                float f = score[i];
                Pair_Scores.Add(f);
            }
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            for (int i = 0; i < PairInfo_Details.Count; i++)
            {
                float g = PairInfo_Details[i];
                float p = Pair_Scores[i];
                mtwriter.WriteLine(g.ToString()+"\t"+p.ToString());
            }
            mtwriter.Close();
        }

    }

    public class ClassificationEvaluationSet : EvaluationSet
    {
        public List<int> PairInfo_Details = new List<int>();
        public List<float[]> Pair_Scores = new List<float[]>();
        public override void Loading_LabelInfo(string[] files)
        {
            FileStream mstream = new FileStream(files[0], FileMode.Open, FileAccess.Read);
            StreamReader mreader = new StreamReader(mstream);

            int Line_Idx = 0;
            while (!mreader.EndOfStream)
            {
                string mline = mreader.ReadLine();
                int label = int.Parse(mline);
                PairInfo_Details.Add(label);
                Line_Idx += 1;
            }
            mreader.Close();
            mstream.Close();
        }

        public override void Save(string scoreFile)
        {
            StreamWriter mtwriter = new StreamWriter(scoreFile);
            for (int i = 0; i < PairInfo_Details.Count; i++)
            {
                StringBuilder sb = new StringBuilder();
                float v = PairInfo_Details[i];
                
                sb.Append(v.ToString() + "\t");
                for (int k = 0; k < Pair_Scores[i].Length; k++)
                {
                    sb.Append(Pair_Scores[i][k].ToString() + "\t");
                }

                mtwriter.WriteLine(sb.ToString());
            }
            mtwriter.Close();
        }

        public override void Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            for (int i = 0; i < args[0]; i++)
            {
                StringBuilder sb = new StringBuilder();
                float[] f = new float[args[1]];
                
                for (int k = 0; k < args[1]; k++)
                {
                    f[k] = score[i * args[1] + k];
                }
                Pair_Scores.Add(f);
            }
        }

        public override void Init()
        {
            Pair_Scores.Clear();
        }
    }



    public class PairInputStream:IDisposable
    {
        public SequenceInputStream qstream = new SequenceInputStream();
        public SequenceInputStream dstream = new SequenceInputStream();

        public static int MAXSEGMENT_BATCH = 40000;
        public static int QUERY_MAXSEGMENT_BATCH = 40000;
        public static int DOC_MAXSEGMENT_BATCH = 40000;

        /********How to transform the qbatch, dbatch and negdbatch into GPU Memory**********/
        public BatchSample_Input GPU_qbatch { get { return qstream.Data; } }
        public BatchSample_Input GPU_dbatch { get { return dstream.Data; } }
        /*************************************************************************************/

        /**************** Associated streams *************/
        public StreamReader srNCEProbDist = null;

        ~PairInputStream()
        {
            Dispose();
        }

        #region For Validation stuff

        EvaluationSet eval = null;
        public void Eval_Init()
        {
            eval.Init();
        }

        public void Eval_Ouput_Batch(float[] score, float[] groundTrue, int[] args)
        {
            eval.Ouput_Batch(score, groundTrue, args);
        }

        public float Eval_Score(out List<string> validationFileLines)
        {
            return eval.Evaluation(out validationFileLines);
        }

        public float Eval_Score_ModelOnlyEvaluationModelOnly(string srcModelPath, string tgtModelPath, out List<string> validationFileLines)
        {
            return EvaluationSet.EvaluationModelOnly(srcModelPath, tgtModelPath, out validationFileLines);
        }

        #endregion

        public void InitFeatureNorm(Normalizer srcNormalizer, Normalizer tgtNormalizer)
        {
            if (srcNormalizer != null)
            {
                if (srcNormalizer.Type == NormalizerType.MIN_MAX)
                {
                    qstream.Init();
                    while (qstream.Fill(ParameterSetting.FEATURE_DIMENSION_QUERY))
                    {
                        srcNormalizer.AnalysisBatch(qstream.Data);
                    }
                    srcNormalizer.AnalysisEnd();
                }
            }
            if (tgtNormalizer != null)
            {
                if (tgtNormalizer.Type == NormalizerType.MIN_MAX)
                {
                    dstream.Init();
                    while (dstream.Fill(ParameterSetting.FEATURE_DIMENSION_DOC))
                    {
                        tgtNormalizer.AnalysisBatch(dstream.Data);
                    }
                    tgtNormalizer.AnalysisEnd();
                }
            }
        }

        /// <summary>
        /// Used by valid input
        /// </summary>
        /// <param name="qFileName"></param>
        /// <param name="dFileName"></param>
        /// <param name="pairFileName"></param>
        public void Load_Validate_PairData(string qFileName, string dFileName, string pairFileName, Evaluation_Type type)
        {
            Load_PairData(qFileName, dFileName, null);

            eval = EvaluationSet.Create(type);

            eval.Loading_LabelInfo(new string[] { pairFileName });
        }
        /// <summary>
        /// Used by training input
        /// </summary>
        /// <param name="qFileName"></param>
        /// <param name="dFileName"></param>
        /// <param name="nceProbDistFile"></param>
        public void Load_Train_PairData(string qFileName, string dFileName, string nceProbDistFile = null)
        {
            Load_PairData(qFileName, dFileName, nceProbDistFile);

            //// We only update feature dimension from train stream on the first fresh kickoff
            //// whenever the feature dimensions have been set or load from models, we will skip the update here
            if (ParameterSetting.FEATURE_DIMENSION_QUERY <= 0 || ParameterSetting.FEATURE_DIMENSION_DOC <= 0)
            {
                ParameterSetting.FEATURE_DIMENSION_QUERY = qstream.Feature_Size;
                ParameterSetting.FEATURE_DIMENSION_DOC = dstream.Feature_Size;

                if (ParameterSetting.MIRROR_INIT)
                {
                    int featureDim = Math.Max(ParameterSetting.FEATURE_DIMENSION_QUERY, ParameterSetting.FEATURE_DIMENSION_DOC);
                    Program.Print(string.Format("Warning! MIRROR_INIT is turned on. Make sure two input sides are on the same feature space, and two models have exactly the same structure. Originally Feature Num Query {0}, Feature Num Doc {1}. Now both aligned to {2}", ParameterSetting.FEATURE_DIMENSION_QUERY, ParameterSetting.FEATURE_DIMENSION_DOC, featureDim));
                    ParameterSetting.FEATURE_DIMENSION_QUERY = featureDim;
                    ParameterSetting.FEATURE_DIMENSION_DOC = featureDim;
                }
            }
        }

        void Load_PairData(string qFileName, string dFileName, string nceProbDistFile)
        {
            CloseAllStreams();
            qstream.get_dimension(qFileName);
            dstream.get_dimension(dFileName);
            if(nceProbDistFile != null)
            {
                this.srNCEProbDist = new StreamReader(nceProbDistFile);
            }

            QUERY_MAXSEGMENT_BATCH = Math.Max(QUERY_MAXSEGMENT_BATCH, qstream.MAXSEQUENCE_PERBATCH);
            DOC_MAXSEGMENT_BATCH = Math.Max(DOC_MAXSEGMENT_BATCH, dstream.MAXSEQUENCE_PERBATCH);
            MAXSEGMENT_BATCH = Math.Max(QUERY_MAXSEGMENT_BATCH, DOC_MAXSEGMENT_BATCH);
        }
        
        public void Init_Batch()
        {
            qstream.Init();
            dstream.Init();
        }

        public bool Next_Batch()
        {
            return Next_Batch(null, null);
        }

        public bool Next_Batch(Normalizer srcNorm, Normalizer tgtNorm)
        {
            if (!qstream.Fill(ParameterSetting.FEATURE_DIMENSION_QUERY) || !dstream.Fill(ParameterSetting.FEATURE_DIMENSION_DOC))
            {
                return false;
            }
            if(srcNorm != null)
                srcNorm.ProcessBatch(qstream.Data);

            if(tgtNorm != null)
                tgtNorm.ProcessBatch(dstream.Data);
            qstream.Data.Batch_In_GPU();
            dstream.Data.Batch_In_GPU();
            return true;
        }

        public void CloseAllStreams()
        {
            //// close existing streams if already opened
            qstream.CloseStream();
            dstream.CloseStream();

            if (this.srNCEProbDist != null)
            {
                this.srNCEProbDist.Close();
                this.srNCEProbDist = null;
            }
        }

        public void Dispose()
        {
            qstream.Dispose();
            dstream.Dispose();
            CloseAllStreams();
        }
    }
}
