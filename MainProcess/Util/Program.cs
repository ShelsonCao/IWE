using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Configuration;
using System.Runtime.InteropServices;

namespace Util
{
    class Program
    {
        static void Main(string[] args)
        {

        }

    }

    public static class NativeMethods
    {
        [StructLayout(LayoutKind.Sequential)]
        internal struct SYSTEM_INFO
        {
            public ushort wProcessorArchitecture;
            public ushort wReserved;
            public uint dwPageSize;
            public IntPtr lpMinimumApplicationAddress;
            public IntPtr lpMaximumApplicationAddress;
            public UIntPtr dwActiveProcessorMask;
            public uint dwNumberOfProcessors;
            public uint dwProcessorType;
            public uint dwAllocationGranularity;
            public ushort wProcessorLevel;
            public ushort wProcessorRevision;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Auto, ExactSpelling = true)]
        internal static extern void GetNativeSystemInfo(ref SYSTEM_INFO lpSystemInfo);
    }

    public static class CmpInfo
    {
        public static int ProcessorCount
        {
            get
            {
                NativeMethods.SYSTEM_INFO lpSystemInfo = new NativeMethods.SYSTEM_INFO();
                NativeMethods.GetNativeSystemInfo(ref lpSystemInfo);
                return (int)lpSystemInfo.dwNumberOfProcessors;
            }
        }
    }

    public class PairGenerator
    {
        private string dicPath = @"../../../../../Data/features/wordlist";
        private string l3gPath = @"../../../../../Data/features/l3g.txt";
        private string corpus = @"../../../../../Data/corpus.txt";

        HashSet<string> dic = new HashSet<string>();
        Dictionary<string, int> l3gDic = new Dictionary<string, int>();
        public int LenCtx = 5;

        public PairGenerator(string corpus, string dicPath, string l3gPath)
        {
            this.corpus = corpus;
            this.dicPath = dicPath;
            this.l3gPath = l3gPath;

            GenWordList(corpus);
            GenL3g(dicPath);
        }

        public void GenWordList(string corpus)
        {
            using (StreamReader sr = new StreamReader(corpus, Encoding.UTF8))
            {
                string line = "";
                while ((line = sr.ReadLine()) != null)
                {
                    string[] terms = line.Split(new string[] { " ", "\t", "\n" }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (string term in terms)
                    {
                        if (!dic.Contains(term))
                        {
                            dic.Add(term);
                        }
                    }
                }
            }
            using (StreamWriter sw = new StreamWriter(dicPath, false, Encoding.UTF8))
            {
                foreach (string term in dic)
                {
                    sw.WriteLine(term);
                }
            }
        }

        private void GenL3g(string dicPath)
        {
            string line = "";
            foreach (string term in dic)
            {
                line = term;
                line = "#" + line + "#";
                for (int i = 0; i < line.Length - 2; i++)   //N = 3
                {
                    if (!l3gDic.ContainsKey(line.Substring(i, 3)))
                    {
                        l3gDic.Add(line.Substring(i, 3), 1);
                    }
                    else
                    {
                        l3gDic[line.Substring(i, 3)]++;
                    }
                }
            }
            try
            {
                using (StreamWriter writer = new StreamWriter(l3gPath, false, Encoding.UTF8))
                {
                    foreach (KeyValuePair<string, int> key in l3gDic)
                    {
                        if (key.Value >= 1)
                        {
                            writer.WriteLine(key.Key);
                        }
                    }
                }
            }
            catch
            {
                Console.WriteLine("Can't read l3g file!");
            }
        }

        public void GenQryTrgFile(string input, string output)
        {
            using (StreamReader reader = new StreamReader(input, Encoding.UTF8))
            {
                using (StreamWriter writer = new StreamWriter(output, false, Encoding.UTF8))
                {
                    int count = 0;
                    string line = null;

                    while ((line = reader.ReadLine()) != null)
                    {
                        if (count % 1000 == 0)
                        {
                            Console.Write("{0}\r", count);
                        }

                        string[] terms = line.Split(' ');

                        if (terms.Length <= 1)
                        {
                            continue;
                        }

                        for (int i = 0; i < terms.Length; i++)
                        {
                            if (!dic.Contains(terms[i]))
                            {
                                continue;
                            }

                            List<string> ctxList = new List<string>();

                            //Order by distance between current word and its context words
                            for (int j = 1; j <= LenCtx; j++)
                            {
                                if (i - j >= 0 && dic.Contains(terms[i - j]))
                                {
                                    ctxList.Add(terms[i - j]);
                                }
                                if (i + j < terms.Length && dic.Contains(terms[i + j]))
                                {
                                    ctxList.Add(terms[i + j]);
                                }
                            }

                            ////ordered by natural sequence
                            //for (int j = i - LenCtx; j <= i + LenCtx; j++)
                            //{
                            //    if (j < 0 || j > terms.Length - 1 || j == i)
                            //    {
                            //        continue;
                            //    }

                            //    ctxList.Add(terms[j]);
                            //}

                            //if (ctxList.Count >= 3)
                            {
                                writer.WriteLine(String.Join(" ", ctxList) + "\t" + terms[i]);
                                count++;
                            }
                        }
                    }
                }
            }
        }

    }
}
