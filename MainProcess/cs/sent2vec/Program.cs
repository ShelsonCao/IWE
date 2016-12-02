using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using jlib;

namespace sent2vec
{
    public enum ModelType
    {
        DSSM,
        CDSSM,
    }

    public class Sent2VecArgs
    {
        [Argument(ArgumentType.Required)]
        public string inSrcModel = "";

        [Argument(ArgumentType.Required)]
        public string inSrcVocab = "";

        public ModelType inSrcModelType = ModelType.DSSM;
        
        public int inSrcMaxRetainedSeqLength = 1;

        [Argument(ArgumentType.Required)]
        public string inTgtModel = "";

        [Argument(ArgumentType.Required)]
        public string inTgtVocab = "";

        public ModelType inTgtModelType = ModelType.DSSM;

        public int inTgtMaxRetainedSeqLength = 1;

        [Argument(ArgumentType.Required)]
        public string inFilename = "";

        [Argument(ArgumentType.Required)]
        public string outFilenamePrefix = "";

        public Sent2VecArgs(string[] args)
        {
            if (!Parser.ParseArgumentsWithUsage(args, this))
            {
                Environment.Exit(-1);
            }
        }

        public void PrintArgs(TextWriter tw)
        {
            tw.WriteLine("==================================");
            tw.WriteLine("# Starting run at {0}", DateTime.Now);
            tw.WriteLine("# Configuration:");
            tw.WriteLine("# --------------------------------");
            tw.WriteLine("inSrcModel={0}", inSrcModel);
            tw.WriteLine("inSrcVocab={0}", inSrcVocab);
            tw.WriteLine("inSrcModelType={0}", inSrcModelType);
            tw.WriteLine("inSrcMaxRetainedSeqLength={0}", inSrcMaxRetainedSeqLength);

            tw.WriteLine("inTgtModel={0}", inTgtModel);
            tw.WriteLine("inTgtVocab={0}", inTgtVocab);
            tw.WriteLine("inTgtModelType={0}", inTgtModelType);
            tw.WriteLine("inTgtMaxRetainedSeqLength={0}", inTgtMaxRetainedSeqLength);

            tw.WriteLine("inFilename={0}", inFilename);
            tw.WriteLine("outFilenamePrefix={0}", outFilenamePrefix);

            tw.WriteLine("==================================");
        }

    }

    public class Program
    {
        private static Dictionary<string, Dictionary<int, float>> ReadWordEmbedFile(string wordEmbedInitFile)
        {
            Console.WriteLine("Reading word embeddings initial file...");
            Dictionary<string, Dictionary<int, float>> dic = new Dictionary<string, Dictionary<int, float>>();

            using (StreamReader reader = new StreamReader(wordEmbedInitFile, Encoding.UTF8))
            {
                string line = reader.ReadLine();
                while ((line = reader.ReadLine()) != null)
                {
                    string[] terms = line.Split(' ');
                    string word = terms[0];
                    Dictionary<int, float> wdIdxFea = new Dictionary<int, float>();

                    for (int i = 1; i < terms.Length - 1; i++)
                    {
                        wdIdxFea.Add(i-1, float.Parse(terms[i]));
                    }

                    dic.Add(word, wdIdxFea);
                }
            }

            return dic;
        }

        public static void Embedding(string inTgtModel, string inTgtVocab, ModelType inTgtModelType, int inTgtMaxRetainedSeqLength,
                              string inFilename, string outFilenamePrefix, int dim, FeatureList featureList)
        {
            Dictionary<string, Dictionary<int, float>> tgtDicWord2Vec = new Dictionary<string, Dictionary<int, float>>();
            int numDic = 0;
            string line = "";

            StreamWriter tfeafp = new StreamWriter(outFilenamePrefix + ".words");

            DNN tgt_dssm = new DNN(inTgtModel, inTgtModelType, inTgtVocab, inTgtMaxRetainedSeqLength);
            if (inTgtModelType == ModelType.DSSM)
                inTgtMaxRetainedSeqLength = 1;

            using (StreamReader sr = new StreamReader(inFilename))
            {
                while ((line = sr.ReadLine()) != null)
                {
                    if (line != "")
                    {
                        numDic++;
                    }
                }
            }

            StreamReader inFile = new StreamReader(inFilename);
            //string line = "";
            int linecnt = 0;

            tfeafp.WriteLine(numDic + " " + dim);
            while ((line = inFile.ReadLine()) != null)
            {
                linecnt++;
                if (0 == linecnt % 1000000) Console.Write("|");
                else if (0 == linecnt % 100000) Console.Write(".");

                string tgtstr = line;

                List<float> tgtvec = null;
                tgtvec = tgt_dssm.Forward(tgtstr, tgtDicWord2Vec, featureList);

                tfeafp.Write(tgtstr + " ");

                foreach (float x in tgtvec)
                    tfeafp.Write(string.Format("{0:0.######} ", x));
                tfeafp.WriteLine();
            }

            if (tfeafp != null) tfeafp.Close();
            Console.WriteLine("total {0} lines processed!", linecnt);
        }

        public static void Main(string[] args)
        {
            try
            {
                Sent2VecArgs o = new Sent2VecArgs(args);
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine(exc.ToString());
                Environment.Exit(0);
            }
        }
    }
}
