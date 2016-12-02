using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace jlib
{
    public class Batch
    {
        private int m_nFeatureDim = 0;
        private List<int> m_rgFeaIdx = new List<int>();
        private List<float> m_rgFeaVal = new List<float>();
        private List<int> m_rgSampleIdx = new List<int>();
        private List<int> m_rgSegIdx = new List<int>();

        public int BatchSize { get { return m_rgSampleIdx.Count; } }
        public int FeatureDim { get { return m_nFeatureDim; } set { m_nFeatureDim = value; } }
        public int ElementSize { get { return m_rgFeaIdx.Count; } }
        public int SegSize { get { return m_rgSegIdx.Count; } }

        public Batch() { }

        public void Clear()
        {
            m_nFeatureDim = 0;
            m_rgFeaIdx.Clear(); m_rgFeaVal.Clear(); m_rgSampleIdx.Clear();
            m_rgSegIdx.Clear();
        }

        /// <summary>
        /// load a list of feature-value pair
        /// </summary>
        /// <param name="fvs"></param>
        public int LoadSample(Dictionary<int, double> fvs)
        {
            int nMaxFid = 0;
            int sid = (BatchSize == 0) ? 0 : m_rgSampleIdx[BatchSize - 1];
            m_rgSampleIdx.Add(sid + fvs.Count);
            foreach (KeyValuePair<int, double> fv in fvs)
            {
                m_rgFeaIdx.Add(fv.Key);
                m_rgFeaVal.Add((float)fv.Value);
                if (fv.Key >= nMaxFid)
                    nMaxFid = fv.Key + 1;
            }
            return nMaxFid;
        }

        public void WriteSample(BinaryWriter bw)
        {
            bw.Write(m_rgSampleIdx.Count);
            bw.Write(m_rgFeaIdx.Count);
            bw.Write(m_nFeatureDim);

            for (int i = 0; i < m_rgSampleIdx.Count; ++i)
                bw.Write(m_rgSampleIdx[i]);

            for (int i = 0; i < m_rgFeaIdx.Count; ++i)
                bw.Write(m_rgFeaIdx[i]);

            for (int i = 0; i < m_rgFeaVal.Count; ++i)
                bw.Write(m_rgFeaVal[i]);
        }

        /// <summary>
        /// Return max feature dimension in this batch
        /// </summary>
        /// <param name="rgDict"></param>
        /// <returns></returns>
        public int LoadSeqSample(List<Dictionary<int, double>> rgDict)
        {
            int nMaxFeatureDimension = 0;

            int sid = (BatchSize == 0) ? 0 : m_rgSampleIdx[BatchSize - 1];
            m_rgSampleIdx.Add(sid + rgDict.Count);

            foreach (Dictionary<int, double> seg in rgDict)
            {
                int wid = (SegSize == 0) ? 0 : m_rgSegIdx[SegSize - 1];
                m_rgSegIdx.Add(wid + seg.Count);
                foreach (KeyValuePair<int, double> fv in seg)
                {
                    m_rgFeaIdx.Add(fv.Key);
                    m_rgFeaVal.Add((float)fv.Value);
                    if (fv.Key >= nMaxFeatureDimension)
                        nMaxFeatureDimension = fv.Key + 1;
                }
            }

            return nMaxFeatureDimension;
        }

        public void WriteSeqSample(BinaryWriter bw)
        {
            bw.Write(SegSize);
            bw.Write(ElementSize);

            for (int i = 0; i < m_rgSampleIdx.Count; ++i)
                bw.Write(m_rgSampleIdx[i]);

            for (int i = 0; i < m_rgSegIdx.Count; ++i)
                bw.Write(m_rgSegIdx[i]);

            for (int i = 0; i < m_rgFeaIdx.Count; ++i)
                bw.Write(m_rgFeaIdx[i]);

            for (int i = 0; i < m_rgFeaVal.Count; ++i)
                bw.Write(m_rgFeaVal[i]);
        }
    }

}
