using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.RegularExpressions;
using System.IO;

namespace jlib
{
    public enum FeatureType
    {
        infl,
        root,
        l3g,
    }

    public class FeatureList
    {
        public bool l3g = false;
        public bool root = false;
        public bool infl = false;
    }

    public class InflModel
    {
        private static InflModel ins = null;

        public static string oriListpath = @"../../../../../Data/features/oriList.txt";
        public static string infl2oriPath = @"../../../../../Data/features/infl2ori.txt";
        public static string wordListPath = @"../../../../../Data/features/wordlist";

        public List<string> oriList;
        public Dictionary<string, int> dic = new Dictionary<string, int>();
        public Dictionary<string, List<string>> dicInfl2Ori;

        public  List<string> wordList;

        public static InflModel getInstance()
        {
            if (ins == null)
            {
                ins = new InflModel();
            }
            return ins;
        }

        private InflModel()
        {
            GetOriList();
            Getinfl2OriDic();
        }

        private void GetOriList()
        {
            string line = "";
            oriList = new List<string>();
            int count = 0;

            using (StreamReader sr = new StreamReader(wordListPath, Encoding.UTF8))
            {
                while ((line = sr.ReadLine()) != null)
                {
                    if (!dic.ContainsKey(line))
                    {
                        dic.Add(line, count++);
                    }
                }
            }

            using (StreamReader sr = new StreamReader(oriListpath, Encoding.UTF8))
            {
                while ((line = sr.ReadLine()) != null)
                {
                    if (dic.ContainsKey(line))
                    {
                        oriList.Add(line);
                    }
                }
            }
        }

        private void Getinfl2OriDic()
        {
            dicInfl2Ori = FileUtils.ReadPairs(infl2oriPath);
            foreach (string ori in oriList)
            {
                if (!dicInfl2Ori.ContainsKey(ori))
                {
                    dicInfl2Ori.Add(ori, new List<string>() { ori });
                }
            }
        }
    }

    public class RootModel
    {
        private static RootModel ins = null;

        public static string rootListpath = @"../../../../../Data/features/rootList.txt";
        public static string word2rootsPath = @"../../../../../Data/features/word2roots.txt";

        public List<string> rootList;
        public Dictionary<string, int> dic = new Dictionary<string, int>();
        public Dictionary<string, List<string>> dicWord2Roots;

        public static RootModel getInstance()
        {
            if (ins == null)
            {
                ins = new RootModel();
            }
            return ins;
        }

        private RootModel()
        {
            GetRootList();
            GetWord2RootsDic();
        }

        private void GetRootList()
        {
            int count = 0;
            string line = "";

            using (StreamReader reader = new StreamReader(rootListpath, Encoding.UTF8))
            {
                while ((line = reader.ReadLine()) != null)
                {
                    dic.Add(line, count++);
                }
            }

            rootList = new List<string>(dic.Keys);
        }

        private void GetWord2RootsDic()
        {
            dicWord2Roots = FileUtils.ReadPairs(word2rootsPath);
        }
    }

    public static class FileUtils
    {
        public static Dictionary<string, List<string>> ReadPairs(string path)
        {
            Dictionary<string, List<string>> dic = new Dictionary<string, List<string>>();
            string word = "";
            List<string> listRoot;

            using (StreamReader reader = new StreamReader(path, Encoding.UTF8))
            {
                string line = "";
                while ((line = reader.ReadLine()) != null)
                {
                    string[] terms = line.Split('\t');
                    word = terms[0];

                    string[] roots = terms[1].Split(',');
                    listRoot = new List<string>();
                    foreach (string str in roots)
                    {
                        listRoot.Add(str);
                    }
                    dic.Add(word, listRoot);
                }
            }

            return dic;
        }
    }

    public static class TextUtils
    {
        public static Regex s_NonAscii_ControlChar = new Regex(@"[^\u0020-\u007F]", RegexOptions.Compiled);
        public static Regex s_NonAscii = new Regex(@"[^\u0000-\u007F]", RegexOptions.Compiled);
        private static Regex n_rg1 = new Regex("[^0-9a-z]", RegexOptions.Compiled);
        private static Regex n_rg2 = new Regex(@"\s+", RegexOptions.Compiled);
        private static Regex s_puncReplace0 = new Regex(@"[^\d\w_'.&\s\-]", RegexOptions.Compiled); // Replace punc with space
        private static Regex s_puncReplace1 = new Regex(@"[^\d\w\s]", RegexOptions.Compiled); // Replace punc with space

        /// <summary>
        /// Split the given string into an array of tokens, using whitespace to delimit each token.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static string[] TokenizeToArray(string s)
        {
            string[] rgs = new string[TokenCount(s)];
            int iString = 0;
            int iPos = 0;
            int iStartPos = -1;
            while (true)
            {
                while (iPos < s.Length && char.IsWhiteSpace(s[iPos]))
                    ++iPos;
                if (iPos >= s.Length)
                    break;
                iStartPos = iPos++;
                while (iPos < s.Length && !char.IsWhiteSpace(s[iPos]))
                    ++iPos;
                rgs[iString++] = s.Substring(iStartPos, iPos - iStartPos);
            }
            return rgs;
        }

        /// <summary>
        /// Find the number of whitespace delimited tokens in the string.
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static int TokenCount(string s)
        {
            int iCount = 0;
            int i = 0;
            while (true)
            {
                while (i < s.Length && char.IsWhiteSpace(s[i]))
                    ++i;
                if (i >= s.Length)
                    break;
                ++iCount;
                ++i;
                while (i < s.Length && !char.IsWhiteSpace(s[i]))
                    ++i;
            }
            return iCount;
        }

        public static string SimpleNorm(string s)
        {
            string ss = s.ToLower();
            ss = ss.Trim();
            return ss;
        }

        public static string Norm(string s, int normType)
        {
            if (normType == 0)
                return s.ToLower();
            else if (normType == 1)
                return N1Normalize(s);
            else if (normType == 2)
                return N2Normalize(s);

            return s;
        }

        /// <summary>
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        public static string N1Normalize(string s)
        {
            string ss = s.ToLower();
            Regex rg1 = n_rg1;
            ss = rg1.Replace(ss, " ");
            Regex rg2 = n_rg2;
            ss = rg2.Replace(ss, " ");
            ss = ss.Trim();

            return ss;
        }

        /// <summary>
        /// remove punctuations only
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public static string N2Normalize(string input)
        {
            input = s_puncReplace1.Replace(input, " ");
            input = input.ToLower().Trim();
            input = n_rg2.Replace(input, " ");

            return input;
        }

        static bool IsNumber(string s)
        {
            double d;
            return double.TryParse(s, out d);
        }

        /// <summary>
        /// convert a string to word-freq pairs
        /// </summary>
        /// <param name="s"></param>
        /// <returns></returns>
        static Dictionary<string, double> String2WordFreqPairs(string s)
        {
            Dictionary<string, double> dict = new Dictionary<string, double>();

            string[] rgs = TokenizeToArray(s);

            if (rgs.Length == 0)
                return dict;

            for (int i = 0; i < rgs.Length; ++i)
            {
                if (!dict.ContainsKey(rgs[i]))
                {
                    dict.Add(rgs[i], 1);
                }
                else
                    dict[rgs[i]]++;
            }

            return dict;
        }
        
        /// <summary>
        /// convert input string to letter-n-gram vector
        /// </summary>
        /// <param name="s">input string</param>
        /// <param name="N">ngram</param>
        /// <returns></returns>
        static Dictionary<string, double> String2L3g(string s, int N)
        {
            Dictionary<string, double> wfs = new Dictionary<string, double>();

            //int id = 0;
            foreach (string w in TokenizeToArray(s))
            {
                string src = "#" + w + "#";
                for (int i = 0; i <= src.Length - N; ++i)
                {
                    string l3g = src.Substring(i, N);
                    if (wfs.ContainsKey(l3g))
                        wfs[l3g]++;
                    else
                        wfs.Add(l3g, 1);
                }
            }
            return wfs;
        }

        public static List<Dictionary<int, double>> String2MatrixFromInitValue(string s, int nMaxLength, Dictionary<string, Dictionary<int, double>> dicWord2Vec)
        {

            List<Dictionary<int, double>> rgWfs = new List<Dictionary<int, double>>();
            string[] rgw = TokenizeToArray(s);

            if (nMaxLength == 1 && rgw.Length != 1)     //it is not supported by DSSM (nMaxLength == 1)
            {
                return rgWfs;
            }

            foreach (string word in rgw)
            {
                //Dictionary<int, double> d = new Dictionary<int, double>();
                if (dicWord2Vec.ContainsKey(word))
                {
                    //d = dicWord2Vec[word];
                    rgWfs.Add(dicWord2Vec[word]);
                }
                //rgWfs.Add(d);
            }

            return rgWfs;
        }

        /// <summary>
        /// convert input string to letter-n-gram sequence, each word is a letter-n-gram vector
        /// </summary>
        /// <param name="s">input string</param>
        /// <param name="v">vocab</param>
        /// <param name="N">ngram</param>
        /// <param name="nMaxLength">max length</param>
        /// <returns></returns>
        /// <summary>
        /// convert input string to letter-n-gram sequence, each word is a letter-n-gram vector
        /// </summary>
        /// <param name="s">input string</param>
        /// <param name="v">vocab</param>
        /// <param name="N">ngram</param>
        /// <param name="nMaxLength">max length</param>
        /// <returns></returns>
        public static List<Dictionary<string, double>> String2FeatStrSeq(string s, int N, int nMaxLength, FeatureType feaType)
        {
            List<Dictionary<string, double>> rgWfs = new List<Dictionary<string, double>>();

            string[] rgw = TokenizeToArray(s);

            for (int i = 0; i < Math.Min(rgw.Length, nMaxLength - 1); ++i)
            {
                switch (feaType)
                {
                    case FeatureType.l3g:
                        {
                            rgWfs.Add(String2L3g(rgw[i], N));
                            break;
                        }
                    case FeatureType.root:
                        {
                            RootModel rootModelIns = RootModel.getInstance();
                            rgWfs.Add(String2Root(rgw[i], rootModelIns.dicWord2Roots));
                            break;
                        }
                    case FeatureType.infl:
                        {
                            InflModel inflModelIns = InflModel.getInstance();
                            rgWfs.Add(String2Root(rgw[i], inflModelIns.dicInfl2Ori));
                            break;
                        }
                }
            }

            Dictionary<string, double> dict = new Dictionary<string, double>();
            for (int i = nMaxLength - 1; i < rgw.Length; ++i)
            {
                Dictionary<string, double> tmp_dict = null;
                switch (feaType)
                {
                    case FeatureType.l3g:
                        {
                            tmp_dict = String2L3g(rgw[i], N);
                            break;
                        }
                    case FeatureType.root:
                        {
                            RootModel rootModelIns = RootModel.getInstance();
                            tmp_dict = String2Root(rgw[i], rootModelIns.dicWord2Roots);
                            break;
                        }
                    case FeatureType.infl:
                        {
                            InflModel inflModelIns = InflModel.getInstance();
                            tmp_dict = String2Root(rgw[i], inflModelIns.dicInfl2Ori);
                            break;
                        }
                }

                foreach (KeyValuePair<string, double> kv in tmp_dict)
                {
                    if (dict.ContainsKey(kv.Key))
                        dict[kv.Key] += kv.Value;
                    else
                        dict.Add(kv.Key, kv.Value);
                }
            }
            if (dict.Count > 0)
                rgWfs.Add(dict);

            return rgWfs;
        }

        private static Dictionary<string, double> String2Root(string s, Dictionary<string,List<string>> dic)
        {
            Dictionary<string, double> wfs = new Dictionary<string, double>();

            //int id = 0;
            foreach (string w in TokenizeToArray(s))
            {
                if (dic.ContainsKey(w))
                {
                    foreach (string root in dic[w])
                    {
                        if (!wfs.ContainsKey(root))
                        {
                            wfs.Add(root, 1);
                        }
                        else
                        {
                            wfs[root]++;
                        }
                    }
                }
            }
            return wfs;
        }

        public static List<Dictionary<int, double>> StrFreq2IdFreq(List<Dictionary<string, double>> strFeqList, Vocab v, int pos)
        {
            List<Dictionary<int, double>> ans = new List<Dictionary<int, double>>(strFeqList.Count);
            int id;
            foreach (var inpDict in strFeqList)
            {
                Dictionary<int, double> dict = new Dictionary<int, double>();
                foreach (var kvp in inpDict)
                {
                    if ((id = v[kvp.Key]) >= 0)
                    {
                        dict[id + pos] = kvp.Value;
                    }
                }
                ans.Add(dict);
            }
            return ans;
        }


        public static Dictionary<int, double> MergeList(List<Dictionary<int, double>> list)
        {
            Dictionary<int, double> dic = new Dictionary<int, double>();

            for (int i = 0; i < list.Count; i++)
            {
                foreach (int idx in list[i].Keys)
                {
                    if (!dic.ContainsKey(idx))
                    {
                        dic.Add(idx, list[i][idx]);
                    }
                    else
                    {
                        dic[idx]++;
                    }
                }
            }
            return dic;
        }

        public static List<Dictionary<int, double>> StrFreq2IdFreq(List<Dictionary<string, double>> strFeqList, FeatureType featureType, int pos, ref int count)
        {
            Dictionary<string, int> dic = new Dictionary<string, int>();

            switch (featureType)
            {
                case FeatureType.root:
                    {
                        RootModel ins = RootModel.getInstance();
                        dic = ins.dic;
                        count = dic.Count;
                        break;
                    }
                case FeatureType.infl:
                    {
                        InflModel ins = InflModel.getInstance();
                        dic = ins.dic;
                        count = dic.Count;
                        break;
                    }
            }


            List<Dictionary<int, double>> ans = new List<Dictionary<int, double>>(strFeqList.Count);
            int id;
            foreach (var inpDict in strFeqList)
            {
                Dictionary<int, double> dict = new Dictionary<int, double>();
                foreach (var kvp in inpDict)
                {
                    if (dic.ContainsKey(kvp.Key) && (id = dic[kvp.Key]) >= 0)
                    {
                        dict[id + pos] = kvp.Value;
                    }
                }
                ans.Add(dict);
            }
            return ans;
        }

        /// <summary>
        /// concatenate 2 feature vectors
        /// </summary>
        /// <param name="tgt"></param>
        /// <param name="src"></param>
        /// <param name="posBias"></param>
        public static void FeatureConcate(Dictionary<int, double> tgt, Dictionary<int, double> src, int posBias)
        {
            foreach (KeyValuePair<int, double> item in src)
            {
                tgt.Add(item.Key + posBias, item.Value);
            }
        }


        /// <summary>
        /// convert vector to a string of key:value
        /// </summary>
        /// <param name="vec"></param>
        /// <returns></returns>
        public static string Vector2String(Dictionary<int, double> vec)
        {
            string str = "";
            bool bFirst = true;
            foreach (KeyValuePair<int, double> wf in vec)
            {
                if (bFirst)
                {
                    str = wf.Key + ":" + (float)wf.Value;
                    bFirst = false;
                }
                else
                    str = str + " " + wf.Key + ":" + (float)wf.Value;
            }
            return str;
        }

        /// <summary>
        /// convert key:value string to vector
        /// </summary>
        /// <param name="str">key:value string</param>
        /// <returns>sparse vector</returns>
        public static Dictionary<int, double> String2Vector(string str)
        {
            Dictionary<int, double> vec = new Dictionary<int, double>();

            string[] rgs = TokenizeToArray(str);
            for (int i = 0; i < rgs.Length; ++i)
            {
                string[] rgw = rgs[i].Split(':');
                int key = int.Parse(rgw[0]);
                double val = double.Parse(rgw[1]);
                if (vec.ContainsKey(key))
                    vec[key] += val;
                else
                    vec.Add(key, val);
            }

            return vec;
        }

        /// <summary>
        /// convert matrix to a string of key:value with # as separator btw vector
        /// </summary>
        /// <param name="mt"></param>
        /// <returns></returns>
        public static string Matrix2String(List<Dictionary<int, double>> mt)
        {
            string s = "";
            bool bFirst = true;
            foreach (Dictionary<int, double> vec in mt)
            {
                if (bFirst)
                {
                    s = Vector2String(vec);
                    bFirst = false;
                }
                else
                    s = s + "#" + Vector2String(vec);
            }
            return s;
        }

        /// <summary>
        /// convert a string of key:value with # as separator btw vector to a matrix
        /// </summary>
        /// <param name="s">a string of key:value with # as separator btw vector</param>
        /// <returns>matrix</returns>
        public static List<Dictionary<int, double>> String2Matrix(string s)
        {
            List<Dictionary<int, double>> mt = new List<Dictionary<int, double>>();
            string[] rgv = s.Split('#');

            for (int i = 0; i < rgv.Length; ++i)
                mt.Add(String2Vector(rgv[i]));

            return mt;
        }
    }

}
