using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Security.Cryptography;

using jlib;

namespace ComputeLogPD
{
    public class ComputeLogPDArgs
    {
        [Argument(ArgumentType.Required)]
        public string inFile = "";

        [Argument(ArgumentType.Required)]
        public string outFile = "";

        [Argument(ArgumentType.Required)]
        public int ColIndex = 1;

        [Argument(ArgumentType.Required)]
        public double Scale = 0.75;

        public ComputeLogPDArgs(string[] args)
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
            tw.WriteLine("inFile={0}", inFile);
            tw.WriteLine("outFile={0}", outFile);
            tw.WriteLine("ColIndex={0} [start from 0]", ColIndex);
            tw.WriteLine("Scale={0} [>=0]", Scale);
            tw.WriteLine("==================================");
        }

    }

    public class Program
    {
        static byte CalculateMD5Hash(string input)
        {
            MD5 md5 = System.Security.Cryptography.MD5.Create();
            byte[] inputBytes = System.Text.Encoding.ASCII.GetBytes(input);
            byte[] hash = md5.ComputeHash(inputBytes);
            return hash[0];
        }

        static void LargeScaleComputeLogPD(string inTsv, int DColumn, double scale, int binNum, string outTsv)
        {
            TextReader inf = new StreamReader(inTsv);
            int colidx = DColumn;

            List<Dictionary<string, float>> ht_list = new List<Dictionary<string, float>>();
            for (int i = 0; i < binNum; i++)
            {
                ht_list.Add(new Dictionary<string, float>());
            }

            string line = null;
            int icnt = 0;
            int ucnt = 0;
            while (true)
            {
                line = inf.ReadLine();
                if (line == null) break;

                icnt++;
                if (0 == (icnt % 10000000))
                    Console.Write("|");
                else if (0 == (icnt % 1000000))
                    Console.Write(".");

                string[] cols = line.Split('\t');
                if (cols.Length <= colidx)
                {
                    throw new Exception("cols.Length < colidx at line " + icnt.ToString());
                }
                string key = cols[colidx].Trim();

                int keyIdx = CalculateMD5Hash(key) % binNum;

                if (!ht_list[keyIdx].ContainsKey(key))
                {
                    ht_list[keyIdx][key] = 1.0f;
                    ucnt += 1;
                }
                else
                    ht_list[keyIdx][key] += 1.0f;
            }
            Console.WriteLine("");
            Console.WriteLine("total {0} lines, unique {1} lines", icnt, ucnt);
            inf.Close();

            double denom = (double)icnt;
            if (scale > -0.0001)
            {
                Console.WriteLine("re-scale by {0}", scale);
                denom = 0;
                foreach (Dictionary<string, float> ht in ht_list)
                {
                    List<string> htkeys = ht.Keys.ToList();
                    foreach (string key in htkeys)
                    {
                        double x = Math.Pow(ht[key], scale);
                        denom += x;
                        ht[key] = (float)x;
                    }
                    htkeys = null;
                }
            }
            double logdenom = Math.Log(denom);

            double Ent = 0;
            foreach (Dictionary<string, float> ht in ht_list)
            {
                foreach (string key in ht.Keys)
                {
                    double logp = Math.Log(ht[key]) - logdenom;
                    Ent += Math.Exp(logp) * logp;
                }
            }
            Console.WriteLine("Entropy {0}", -Ent);

            inf = new StreamReader(inTsv);
            TextWriter dis = new StreamWriter(outTsv);
            icnt = 0;
            while (true)
            {
                line = inf.ReadLine();
                if (line == null) break;

                icnt++;
                if (0 == (icnt % 10000000)) Console.Write("|");
                else if (0 == (icnt % 1000000)) Console.Write(".");

                string[] cols = line.Split('\t');
                if (cols.Length <= colidx)
                {
                    throw new Exception("cols.Length < colidx at line " + icnt.ToString());
                }
                string key = cols[colidx].Trim();

                int keyIdx = CalculateMD5Hash(key) % binNum;

                double logp = Math.Log(ht_list[keyIdx][key]) - logdenom;
                dis.WriteLine("{0:f6}", logp);
            }
            inf.Close();
            dis.Close();
            Console.WriteLine("");
        }

        static void ComputeLogPD(string inTsv, int DColumn, double scale, string outTsv)
        {
            LargeScaleComputeLogPD(inTsv, DColumn, scale, 1, outTsv);
        }

        public static void Main(string[] args)
        {
            try
            {
                ComputeLogPDArgs o = new ComputeLogPDArgs(args);
                ComputeLogPD(o.inFile, o.ColIndex, Math.Max(0.0, o.Scale), o.outFile);
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine(exc.ToString());
                Environment.Exit(0);
            }
        }
    }
}
