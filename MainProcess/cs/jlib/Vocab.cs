using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace jlib
{
    /// <summary>
    /// Simple vocab
    /// </summary>
    [Serializable]
    public class Vocab
    {
        public Vocab(Vocab v)
        {
            m_list = new List<string>(v.m_list);
            m_dict = new Dictionary<string, int>(v.m_dict);
            m_fLocked = v.m_fLocked;
            m_iUnk = v.m_iUnk;
        }

        public Vocab()
            : this(true)
        {
        }

        public Vocab(bool fIncludeNull)
        {
            if (fIncludeNull)
            {
                m_list.Add("[NULL]");
                m_dict["[NULL]"] = 0;
            }
        }

        public void Write(string sFilename)
        {
            using (StreamWriter sw = new StreamWriter(sFilename, false, Encoding.Unicode))
            {
                for (int i = 0; i < Count; ++i)
                {
                    sw.WriteLine("{0}", m_list[i]);
                }
            }
        }

        public void Read(string sFilename)
        {
            m_list.Clear();
            m_dict.Clear();

            string sLine = "";
            using (StreamReader sr = new StreamReader(sFilename))
            {
                while (null != (sLine = sr.ReadLine()))
                {
                    // string sTok = sLine;
                    string sTok = sLine.Split('\t')[0];
                    m_dict[sTok] = m_list.Count;
                    m_list.Add(sTok);
                }
            }
        }

        public int Lookup(string s)
        {
            int iRet;
            if (m_dict.TryGetValue(s, out iRet))
            {
                return iRet;
            }
            return m_iUnk;
        }

        public int Encode(string s)
        {
            if (m_fLocked)
                throw new Exception("Can't encode on locked vocab!");
            int iRet = 0;
            if (m_dict.TryGetValue(s, out iRet))
                return iRet;
            if (s == m_strUnk)
                return m_iUnk;

            iRet = m_list.Count;
            m_list.Add(s);
            m_dict[s] = iRet;
            return iRet;
        }

        public int Unk { get { return m_iUnk; } set { m_iUnk = value; } }

        public int VocabSize { get { return m_list.Count; } }

        public bool Locked { get { return m_fLocked; } }
        public void Lock() { m_fLocked = true; }
        public void Unlock() { m_fLocked = false; }

        public int this[string s]
        {
            get
            {
                if (m_fLocked)
                    return Lookup(s);
                return Encode(s);
            }
            set
            {
                int iOldValue;
                if (m_dict.TryGetValue(s, out iOldValue))
                {
                    if (iOldValue != value)
                    {
                        throw new Exception("Can't remap existing vocab item!");
                    }
                    return;
                }
                if (m_fLocked)
                {
                    throw new Exception("Vocabulary is locked; cannot add entries.");
                }
                while (m_list.Count <= value)
                {
                    m_list.Add(null);
                }
                m_list[value] = s;
                m_dict[s] = value;
            }
        }

        public string this[int i]
        {
            get
            {
                if (i == Unk)
                    return m_strUnk;
                return m_list[i];
            }
        }

        public IEnumerable<string> Words { get { return m_list; } }

        public int Count
        {
            get { return m_list.Count; }
        }

        Dictionary<string, int> m_dict = new Dictionary<string, int>();
        List<string> m_list = new List<string>();
        protected int m_iUnk = -1;
        protected bool m_fLocked = false;
        protected string m_strUnk = "<UNK>";
    }
}
