using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.Security.Cryptography;
using jlib;
using sent2vec;
using Util;
using System.Runtime.InteropServices;
using DSMlib; 

namespace WordHash
{
    static class ExternalShuffle
    {
        private static int BLOCK_SIZE = 1024 * 1024 * 512; //512MB

        public static void Shuffle(string infile, string outfile, string dir)
        {
            if (string.IsNullOrEmpty(dir))
            {
                dir = Path.GetTempPath();
            }
            if(!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);                    
            }
            if(File.Exists(outfile))
            {
                File.Delete(outfile);
            }
            string mytmpPath = dir + "\\shuffleDir" + DateTime.Now.GetHashCode() + "\\";
            Console.WriteLine(mytmpPath);
            DateTime start = DateTime.Now;
            Console.WriteLine("[{0}] External sort started", start);
            int lineCount = 0;
            if (Directory.Exists(mytmpPath))
            {
                Directory.Delete(mytmpPath, true);
            }
            Directory.CreateDirectory(mytmpPath);
            long fileSize = new FileInfo(infile).Length;
            int numFile = (int)((fileSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

            lineCount = Split(infile, mytmpPath, numFile);
            Merge(outfile, mytmpPath, ".seg", numFile);
            Directory.Delete(mytmpPath, true);
            Console.WriteLine("[{0}] External sort finished. {1} lines. Time used: {2}", DateTime.Now, lineCount, DateTime.Now - start);
        }

        public static void Merge(string output, string dir, string suffix, int numFile)
        {
            if (File.Exists(output))
            {
                File.Delete(output);
            }

            for (int i = 0; i < numFile; i++)
            {
                Console.Write("\r[{0}] Merging with file: {1}/{2}", DateTime.Now, i + 1, numFile);

                FileStream TempStreamB = new FileStream(dir + i + suffix, FileMode.Open);
                FileStream AddStreamB = new FileStream(output, FileMode.Append);

                long reLen = TempStreamB.Length;
                byte[] tmp;

                while (reLen > BLOCK_SIZE)
                {
                    tmp = new byte[BLOCK_SIZE];
                    TempStreamB.Read(tmp, 0, BLOCK_SIZE);
                    AddStreamB.Write(tmp, 0, BLOCK_SIZE);

                    reLen = reLen - BLOCK_SIZE;
                }

                int reL = (int)reLen;
                tmp = new byte[reL];
                TempStreamB.Read(tmp, 0, reL);
                AddStreamB.Write(tmp, 0, reL);

                AddStreamB.Close();
                TempStreamB.Close();

                File.Delete(dir + i + suffix);
            }
            Console.WriteLine();
        }

        public static int Split(string fileName, string dir, int numFile)
        {
            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            Console.WriteLine("[{0}] Splitting begains", DateTime.Now);
            List<StreamWriter> writerList = new List<StreamWriter>();
            for (int i = 0; i < numFile; i++)
            {
                writerList.Add(new StreamWriter(dir + "\\" + i + ".seg"));
            }
            int cnt = 0;
            using (StreamReader reader = new StreamReader(fileName))
            {
                string line;
                //random
                Random rand = new Random(7);
                int rdm;

                while (null != (line = reader.ReadLine()))
                {
                    if ((cnt) % 1000000 == 0)
                    {
                        Console.Write("\r[{0}] {1:#,0} lines processed", DateTime.Now, cnt);
                    }
                    rdm = rand.Next(numFile);
                    writerList[rdm].WriteLine(line);

                    //writerList[cnt % numFile].WriteLine(line);        //don't use random number
                    cnt++;
                }
                Console.WriteLine("");
                //close
                for (int i = 0; i < numFile; i++)
                {
                    writerList[i].Close();
                }
            }
            return cnt;
        }

    }
    public class Program
    {
        /// <summary>
        /// convert seq fea file to bin file.
        /// </summary>
        /// <param name="inFile">input seq fea file</param>
        /// <param name="outFile">output bin file</param>
        static void SeqFea2Bin(string inFile, int BatchSize, string outFile)
        {
            string[] terms = inFile.Split('/');
            string suffix = terms[terms.Length - 1];

            string outputDir = @"../../../../../Data/tmp/";
            int nThreads = CmpInfo.ProcessorCount;

            int totalLine = ExternalShuffle.Split(inFile, outputDir, nThreads);

            List<int> nMaxFeatureNumPerBatch = new List<int>();
            List<int> nMaxFeatureDimension = new List<int>();
            List<int> featureDimension = new List<int>();
            List<int> nMaxSegmentSize = new List<int>();
            List<int> nLine = new List<int>();

            for (int i = 0; i < nThreads; i++)
            {
                nMaxFeatureNumPerBatch.Add(0);
                nMaxFeatureDimension.Add(0);
                featureDimension.Add(0);
                nMaxSegmentSize.Add(0);
                nLine.Add(0);
            }

            Parallel.For(0, nThreads, id =>
                {
                    BinaryWriter bw = new BinaryWriter(File.Open(outputDir + id + suffix, FileMode.Create));
                    string sLine = ""; //int nLine = 0;
                    List<Dictionary<int, double>> rgWfs = new List<Dictionary<int, double>>();

                    Batch batch = new Batch();

                    using (StreamReader sr = new StreamReader(outputDir + id + ".seg"))
                    {
                        while ((sLine = sr.ReadLine()) != null)
                        {
                            nLine[id]++; if (nLine[id] % 10000 == 0) Console.Write("{0}\r", nLine[id]);

                            rgWfs = TextUtils.String2Matrix(sLine.Trim());

                            // binary output
                            if (batch.BatchSize == BatchSize)
                            {
                                if (batch.ElementSize > nMaxFeatureNumPerBatch[id])
                                    nMaxFeatureNumPerBatch[id] = batch.ElementSize;
                                // batch.FeatureDim = nMaxFeatureId;
                                batch.WriteSeqSample(bw);
                                batch.Clear();
                            }
                            featureDimension[id] = batch.LoadSeqSample(rgWfs);
                            if (featureDimension[id] > nMaxFeatureDimension[id])
                                nMaxFeatureDimension[id] = featureDimension[id];
                            if (batch.SegSize > nMaxSegmentSize[id])
                                nMaxSegmentSize[id] = batch.SegSize;
                        }
                    }

                    // binary output
                    if (batch.BatchSize > 0)
                    {
                        if (batch.ElementSize > nMaxFeatureNumPerBatch[id])
                            nMaxFeatureNumPerBatch[id] = batch.ElementSize;
                        // batch.FeatureDim = nMaxFeatureId;
                        batch.WriteSeqSample(bw);
                        batch.Clear();
                    }
                    bw.Close();
                    File.Delete(outputDir + id + ".seg");
                });

            ExternalShuffle.Merge(outFile, outputDir, suffix, nThreads);
            BinaryWriter bwTail = new BinaryWriter(File.Open(outFile, FileMode.Append));

            totalLine = nLine.Sum();

            bwTail.Write(nMaxFeatureDimension.Max()); bwTail.Write(totalLine); bwTail.Write(nMaxSegmentSize.Max()); bwTail.Write(nMaxFeatureNumPerBatch.Max());
            bwTail.Write(BatchSize); // part of change on 2/19/2014. Write the batch size at the end. Used to check consistency in training.
            bwTail.Close();
        }

        static int[] Discretize(double[] oriValueList, double minValue, double maxValue, int numFea)
        {
            double range = maxValue - minValue;
            int[] newValueList = new int[oriValueList.Length];

            //for (int i = 0; i < oriValueList.Length;i++ )
            Parallel.For(0, oriValueList.Length, i =>
            {
                newValueList[i] = (int)Math.Ceiling((oriValueList[i] - minValue) * (numFea / range)) - 1;
                newValueList[i] = newValueList[i] + i * numFea;
            });
            return newValueList;
        }

        static int Discretize(double oriValue, double minValue, double maxValue, int numFea)
        {
            double range = maxValue - minValue;
            int idx = (int)Math.Ceiling((oriValue - minValue) * (numFea / range)) - 1;
            return idx;
        }

        private static Dictionary<string, Dictionary<int, double>> ReadWordEmbedFile(string wordEmbedInitFile)
        {
            Console.WriteLine("Reading word embeddings initial file...");
            Dictionary<string, Dictionary<int, double>> dic = new Dictionary<string, Dictionary<int, double>>();

            using (StreamReader reader = new StreamReader(wordEmbedInitFile, Encoding.UTF8))
            {
                string line = reader.ReadLine();
                while ((line = reader.ReadLine()) != null)
                {
                    string[] terms = line.Split(' ');
                    string word = terms[0];
                    Dictionary<int, double> wdIdxFea = new Dictionary<int, double>();

                    for (int i = 1; i < terms.Length - 1; i++)
                    {
                        wdIdxFea.Add(i - 1, double.Parse(terms[i]));
                    }
                    dic.Add(word, wdIdxFea);
                }
            }
            return dic;
        }

        public static void Merge(ref List<Dictionary<int, double>> rgWfs, List<Dictionary<int, double>> tmp)
        {
            for (int i = 0; i < tmp.Count; i++)
            {
                foreach (int idx in tmp[i].Keys)
                {
                    rgWfs[i].Add(idx, tmp[i][idx]);
                }
            }
        }

        public static void Pair2SeqFeaBin(string inFile, string vocFile, int nMaxLength, int idx, string outFile, int BatchSize, FeatureList featureList)
        {
            Dictionary<string, Dictionary<int, double>> dicRoot = new Dictionary<string, Dictionary<int, double>>();
            Vocab voc = new Vocab(false);

            if (featureList.l3g == true)
            {
                voc.Read(vocFile); voc.Lock();
            }


            int N = 3;  // letter 3-gram

            string outputDir = @"../../../../../Data/tmp/";
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            int nThreads = CmpInfo.ProcessorCount;
            string suffix = ".bin";

            //for debug
            //nThreads = 1;

            List<int> nMaxFeatureNumPerBatch = new List<int>();
            List<int> nMaxFeatureDimension = new List<int>();
            List<int> featureDimension = new List<int>();
            List<int> nMaxSegmentSize = new List<int>();
            List<int> nBatch = new List<int>();

            for (int i = 0; i < nThreads; i++)
            {
                nMaxFeatureNumPerBatch.Add(0);
                nMaxFeatureDimension.Add(0);
                featureDimension.Add(0);
                nMaxSegmentSize.Add(0);
                nBatch.Add(0);
            }

            int totalLine = ExternalShuffle.Split(inFile, outputDir, nThreads);

            Parallel.For(0, nThreads, id =>
            {
                BinaryWriter bw = new BinaryWriter(File.Open(outputDir + id + suffix, FileMode.Create));
                StreamWriter sw = new StreamWriter(File.Open(outputDir + id + ".tsv", FileMode.Create));
                StringBuilder sb = new StringBuilder();
                Batch batch = new Batch();

                string sLine = ""; int nLine = 0;
                using (StreamReader sr = new StreamReader(outputDir + id + ".seg"))
                {
                    while ((sLine = sr.ReadLine()) != null)
                    {
                        nLine++; if (nLine % 1000 == 0) Console.Write("{0}\r", nLine);
                        sb.Append(sLine + "\n");

                        string labelLine = string.Empty;
                        string[] rgs = sLine.Split('\t');

                        if (rgs.Length <= idx)
                        {
                            throw new Exception("Invalid format in input file! Exactly two fields separated by tabs are expected " + sLine.ToLower());
                        }

                        int pos = 0;

                        List<Dictionary<int, double>> rgWfs = new List<Dictionary<int, double>>();
                        string[] words = TextUtils.TokenizeToArray(rgs[idx]);
                        for (int i = 0; i < words.Length; i++)
                        {
                            rgWfs.Add(new Dictionary<int, double>());
                        }


                        if (featureList.l3g == true)
                        {
                            var featStrFeq = TextUtils.String2FeatStrSeq(rgs[idx], N, nMaxLength, FeatureType.l3g);  // letter N-gram
                            List<Dictionary<int, double>> tmp = TextUtils.StrFreq2IdFreq(featStrFeq, voc, pos);
                            Merge(ref rgWfs, tmp);
                            pos += voc.Count;
                        }
                        if (featureList.root == true)
                        {
                            int count = 0;
                            var featStrFeq = TextUtils.String2FeatStrSeq(rgs[idx], N, nMaxLength, FeatureType.root);  // list of root
                            List<Dictionary<int, double>> tmp = TextUtils.StrFreq2IdFreq(featStrFeq, FeatureType.root, pos, ref count);
                            Merge(ref rgWfs, tmp);
                            pos += count;
                        }
                        if (featureList.infl == true)
                        {
                            int count = 0;
                            var featStrFeq = TextUtils.String2FeatStrSeq(rgs[idx], N, nMaxLength, FeatureType.infl);  // list of inflections
                            List<Dictionary<int, double>> tmp = TextUtils.StrFreq2IdFreq(featStrFeq, FeatureType.infl, pos, ref count);
                            Merge(ref rgWfs, tmp);
                            pos += count;
                        }


                        // binary output
                        if (batch.BatchSize == BatchSize)
                        {
                            if (batch.ElementSize > nMaxFeatureNumPerBatch[id])
                                nMaxFeatureNumPerBatch[id] = batch.ElementSize;
                            // batch.FeatureDim = nMaxFeatureId;
                            batch.WriteSeqSample(bw);
                            batch.Clear();
                            sw.Write(sb);
                            sb = new StringBuilder();
                            nBatch[id]++;
                        }
                        featureDimension[id] = batch.LoadSeqSample(rgWfs);
                        if (featureDimension[id] > nMaxFeatureDimension[id])
                            nMaxFeatureDimension[id] = featureDimension[id];
                        if (batch.SegSize > nMaxSegmentSize[id])
                            nMaxSegmentSize[id] = batch.SegSize;
                    }
                }
                //Console.WriteLine("nLine");

                // binary output
                if (batch.BatchSize > 0)
                {
                    batch.Clear();
                }

                bw.Close();
                sw.Close();
                File.Delete(outputDir + id + ".seg");
            });

            voc.Unlock();

            ExternalShuffle.Merge(outFile, outputDir, suffix, nThreads);
            BinaryWriter bwTail = new BinaryWriter(File.Open(outFile, FileMode.Append));
            
            totalLine = nBatch.Sum() * BatchSize;
            bwTail.Write(nMaxFeatureDimension.Max()); bwTail.Write(totalLine); bwTail.Write(nMaxSegmentSize.Max()); bwTail.Write(nMaxFeatureNumPerBatch.Max());
            bwTail.Write(BatchSize); // part of change on 2/19/2014. Write the batch size at the end. Used to check consistency in training.
            bwTail.Close();

            ExternalShuffle.Merge(ParameterSetting.trainPairTokzNew, outputDir, ".tsv", nThreads);
            if (Directory.Exists(outputDir))
            {
                Directory.Delete(outputDir);
            }
        }

        static void DispHelp()
        {
            Console.WriteLine("WordHash.exe --shuffle inFile outFile");
            Console.WriteLine("WordHash.exe --pair2seqfea inPair srcVoc tgtVoc maxRetainedSeqLength outFileNamePrefix");
            Console.WriteLine("WordHash.exe --seqfea2bin inFea batchSize outBin");
        }

        public static void Main(string[] args)
        {
            try
            {
                if (args.Length < 1)
                {
                    DispHelp();
                }
                else if (args[0].ToLower() == "--pair2seqfea" && args.Length == 7)
                {
                    //Pair2SeqFea(args[1], args[2], args[3], int.Parse(args[4]), args[5], args[6]);        //update by Shelson, Jan. 18
                }
                else if (args[0].ToLower() == "--seqfea2bin" && args.Length == 4)
                {
                    SeqFea2Bin(args[1], int.Parse(args[2]), args[3]);
                }
                else if (args[0].ToLower() == "--shuffle" && (args.Length == 3 || args.Length == 4))
                {
                    ExternalShuffle.Shuffle(args[1], args[2], args.Length == 4 ? args[3] : null);
                }
                else
                {
                    DispHelp();
                }
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine(exc.ToString());
                //Environment.Exit(0);
            }
        }
    }
}
