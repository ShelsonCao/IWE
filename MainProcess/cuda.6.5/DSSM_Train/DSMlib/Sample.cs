using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace DSMlib
{
    public enum NormalizerType
    {
        NONE = 0,
        MIN_MAX = 1
    }

    public class Normalizer
    {
        int FeatureDim = 0;
        public virtual NormalizerType Type
        {
            get
            {
                return NormalizerType.NONE;
            }
        }

        public Normalizer(int featureDim)
        {
            FeatureDim = featureDim;
        }
        public virtual void AnalysisBatch(BatchSample_Input input)
        { }

        public virtual void AnalysisEnd()
        { }

        public virtual void ProcessBatch(BatchSample_Input input)
        { }
        
        public static Normalizer CreateFeatureNormalize(NormalizerType type, int featureSize)
        {
            Normalizer norm = null;
            switch (type)
            {
                case NormalizerType.NONE:
                    norm = new Normalizer(featureSize);
                    break;
                case NormalizerType.MIN_MAX:
                    norm = new MinMaxNormalizer(featureSize);
                    break;
            }
            return norm;
        }
    }

    public class MinMaxNormalizer : Normalizer
    {
        public float[] FeaMin;
        public float[] FeaMax;
        Dictionary<int, int> FeaDict = new Dictionary<int, int>();
        int SampleNum = 0;
        public MinMaxNormalizer(int featureDim)
            : base(featureDim)
        {
            FeaMin = new float[featureDim];
            FeaMax = new float[featureDim];
            for (int i = 0; i < featureDim; i++)
            {
                FeaMin[i] = float.MaxValue;
                FeaMax[i] = float.MinValue;
            }
        }

        public override void AnalysisEnd()
        {
            foreach (int fid in FeaDict.Keys)
            {
                if (FeaDict[fid] < SampleNum)
                {
                    if (FeaMin[fid] > 0)
                        FeaMin[fid] = 0;
                    if (FeaMax[fid] < 0)
                        FeaMax[fid] = 0;
                }
            }
        }

        public override NormalizerType Type
        {
            get
            {
                return NormalizerType.MIN_MAX;
            }
        }

        public override void AnalysisBatch(BatchSample_Input input)
        {
            SampleNum += input.batchsize;
            for (int sample = 0; sample < input.batchsize; ++sample)
            {
                int seg_begin = sample >= 1 ? input.Sample_Idx_Mem[sample - 1] : 0;
                int seg_end = input.Sample_Idx_Mem[sample];

                for (int seg_id = seg_begin; seg_id < seg_end; ++seg_id)
                {
                    int fea_begin = seg_id >= 1 ? input.Seg_Idx_Mem[seg_id - 1] : 0;
                    int fea_end = input.Seg_Idx_Mem[seg_id];

                    for (int fea_id = fea_begin; fea_id < fea_end; ++fea_id)
                    {
                        int fea_key = input.Fea_Idx_Mem[fea_id];
                        float fea_value = input.Fea_Value_Mem[fea_id];
                        
                        if (FeaDict.ContainsKey(fea_key))
                            FeaDict[fea_key] += 1;
                        else
                            FeaDict[fea_key] = 1;

                        if (FeaMin[fea_key] > fea_value)
                            FeaMin[fea_key] = fea_value;
                        if (FeaMax[fea_key] < fea_value)
                            FeaMax[fea_key] = fea_value;
                    }
                }
            }
        }

        public override void ProcessBatch(BatchSample_Input input)
        {
            for (int sample = 0; sample < input.batchsize; ++sample)
            {
                int seg_begin = sample >= 1 ? input.Sample_Idx_Mem[sample - 1] : 0;
                int seg_end = input.Sample_Idx_Mem[sample];

                for (int seg_id = seg_begin; seg_id < seg_end; ++seg_id)
                {
                    int fea_begin = seg_id >= 1 ? input.Seg_Idx_Mem[seg_id - 1] : 0;
                    int fea_end = input.Seg_Idx_Mem[seg_id];

                    for (int fea_id = fea_begin; fea_id < fea_end; ++fea_id)
                    {
                        int fea_key = input.Fea_Idx_Mem[fea_id];
                        float fea_value = input.Fea_Value_Mem[fea_id];
                        float new_fea_value = (fea_value - FeaMin[fea_key]) / (FeaMax[fea_key] - FeaMin[fea_key] + float.Epsilon);
                        input.Fea_Value_Mem[fea_id] = new_fea_value;
                    }
                }
            }
        }

    }

    public class BatchSample_Input:IDisposable
    {
        public int batchsize;
        public int segsize; // the total length of the full segments.
        public int elementsize;
        public int featureDim;

        CudaPieceInt sample_Idx;
        CudaPieceInt seg_Idx;
        CudaPieceInt seg_Margin;
        CudaPieceFloat seg_Len;
        CudaPieceInt fea_Idx;
        CudaPieceFloat fea_Value;

        public int[] Fea_Idx_Mem { get { return fea_Idx.MemPtr; } }
        public float[] Fea_Value_Mem { get { return fea_Value.MemPtr; } }
        public int[] Sample_Idx_Mem { get { return sample_Idx.MemPtr; } }
        public int[] Seg_Idx_Mem { get { return seg_Idx.MemPtr; } }
        public int[] Seg_Margin_Mem { get { return seg_Margin.MemPtr; } }
        public float[] Seg_Len_Mem { get { return seg_Len.MemPtr; } }

        public IntPtr Fea_Idx { get { return fea_Idx.CudaPtr; } }
        public IntPtr Fea_Value { get { return fea_Value.CudaPtr; } }
        public IntPtr Sample_Idx { get { return sample_Idx.CudaPtr; } }
        public IntPtr Seg_Idx { get { return seg_Idx.CudaPtr; } }
        public IntPtr Seg_Margin { get { return seg_Margin.CudaPtr; } }
        public IntPtr Seg_Len { get { return seg_Len.CudaPtr; } }

        
        public BatchSample_Input(int MAX_BATCH_SIZE, int MAXSEQUENCE_PERBATCH, int MAXELEMENTS_PERBATCH)
        {
            sample_Idx = new CudaPieceInt(MAX_BATCH_SIZE, true, true);
            seg_Idx = new CudaPieceInt(MAXSEQUENCE_PERBATCH, true, true);
            seg_Margin = new CudaPieceInt(MAXSEQUENCE_PERBATCH, true, true);
            seg_Len = new CudaPieceFloat(MAXSEQUENCE_PERBATCH, true, true);
            fea_Idx = new CudaPieceInt(MAXELEMENTS_PERBATCH, true, true);
            fea_Value = new CudaPieceFloat(MAXELEMENTS_PERBATCH, true, true);            
        }
        ~BatchSample_Input()
        {
            Dispose();
        }

        /// <summary>
        /// Return the maxFeatureId found in this batch
        /// </summary>
        /// <param name="mreader"></param>
        /// <param name="expectedBatchSize"></param>
        public int Load(BinaryReader mreader, int expectedBatchSize, bool needToReturnFeatureDimInThisBatch = false)
        {
            // read header
            if (ParameterSetting.LoadInputBackwardCompatibleMode == "BOW")
            {
                // backward-compatible mode for "BOW" format
                batchsize = mreader.ReadInt32(); //read int four-byte.                
                segsize = batchsize; // because one seg per batch.
                elementsize = mreader.ReadInt32(); // read the element size.                
                featureDim = mreader.ReadInt32(); //read int four-byte.
                // The element in the old BOW format is equal to the seg in the new SEQ format, which is one sparse feature vector (representing one (bag of) word).
                // so in the old BOW format (where window_size is 1), Sample_Idx is always 1 to batch_size
                for (int i = 0; i < batchsize; ++i)
                {
                    Sample_Idx_Mem[i] = i + 1;  // populate sample idx for the old BOW format.
                }
            }
            else if (ParameterSetting.LoadInputBackwardCompatibleMode == "SEQ")
            {
                // backward-compatible mode for "SEQ" format
                batchsize = mreader.ReadInt32(); //read int four-byte.
                segsize = mreader.ReadInt32();
                elementsize = mreader.ReadInt32(); //read int four-byte.
                featureDim = mreader.ReadInt32(); //read int four-byte.

                for (int i = 0; i < batchsize; i++)  //read sample index.
                {
                    Sample_Idx_Mem[i] = mreader.ReadInt32();
                }
            }
            else
            {
                // no more reading batch_size or feature_dim here any more.
                batchsize = expectedBatchSize;	// this is needed because the last batch may be incomplete and less than ParameterSetting.Batch_Size
                segsize = mreader.ReadInt32();
                elementsize = mreader.ReadInt32(); //read int four-byte.

                for (int i = 0; i < batchsize; i++)  //read sample index.
                {
                    Sample_Idx_Mem[i] = mreader.ReadInt32();
                }
            }

            // update cudaDataPiece sizes
            sample_Idx.Size = batchsize;
            seg_Idx.Size = segsize;
            seg_Len.Size = segsize;
            seg_Margin.Size = segsize;
            fea_Idx.Size = elementsize;
            fea_Value.Size = elementsize;

            // the remaining are the same routine
            int smp_index = 0;
            for (int i = 0; i < segsize; i++)
            {
                Seg_Idx_Mem[i] = mreader.ReadInt32();
                while (Sample_Idx_Mem[smp_index] <= i)
                {
                    smp_index++;
                }
                Seg_Margin_Mem[i] = smp_index;
                Seg_Len_Mem[i] = 0;
            }
            for (int i = 0; i < elementsize; i++)
            {
                Fea_Idx_Mem[i] = mreader.ReadInt32();
            }
            if (needToReturnFeatureDimInThisBatch)
            {
                // if feature dim in this batch is not needed, we may skip this loop for better efficiency
                // return feature dim, which is the max seen feature ID + 1
                this.featureDim = Enumerable.Range(0, elementsize).Max(i => this.Fea_Idx_Mem[i]) + 1;                
            }

            float sum = 0;
            int seg_index = 0;
            if (ParameterSetting.FeatureValueAsInt)
            {
                for (int i = 0; i < elementsize; i++)
                {
                    Fea_Value_Mem[i] = (float)mreader.ReadInt32();
                    while (Seg_Idx_Mem[seg_index] <= i)
                    {
                        Seg_Len_Mem[seg_index] = sum;
                        seg_index++;
                        sum = 0;
                    }
                    sum += Fea_Value_Mem[i];// *Fea_Value[i];
                }
                Seg_Len_Mem[seg_index] = sum; 

            }
            else
            {
                for (int i = 0; i < elementsize; i++)
                {
                    Fea_Value_Mem[i] = mreader.ReadSingle();

                    while (Seg_Idx_Mem[seg_index] <= i)
                    {
                        Seg_Len_Mem[seg_index] = sum;
                        seg_index++;
                        sum = 0;
                    }
                    sum += Fea_Value_Mem[i];// *Fea_Value[i];
                }
                Seg_Len_Mem[seg_index] = sum; 

            }
            return this.featureDim;
        }
        /// <summary>
        /// Usually called by valid/test corpus, remove feature/value pairs whose feature id is beyond that scope defined by the training data.
        /// Should be called on each side as well. 
        /// For efficiency reason, only call this function when necessary
        /// </summary>
        /// <param name="featureSize">the allowed feature size. We need to prune according to this feature size</param>
        public void FilterOOVFeature(int featureSize)
        {
            // Batchsize : will have no change
            // FeatureDim : will have no change
            // segsize : will have no change
            // Sample_Idx : will have no change
            // Seg_Margin : will have no change

            // Seg_Idx : may be changed
            // elementsize : may be reduced
            // Fea_Idx : may be reduced
            // Fea_Value : may be reduced
            // Seg_Len : may be changed
            int reduced_elementsize = 0;
            int seg_id = 0;
            int ele_id = 0;
            for (int sample = 0; sample < batchsize; ++sample)
            {
                for (; seg_id < Sample_Idx_Mem[sample]; ++seg_id)
                {
                    float sum = 0;
                    for (; ele_id < Seg_Idx_Mem[seg_id]; ++ele_id)
                    {
                        if (Fea_Idx_Mem[ele_id] < featureSize)
                        {
                            Fea_Idx_Mem[reduced_elementsize] = Fea_Idx_Mem[ele_id];
                            Fea_Value_Mem[reduced_elementsize] = Fea_Value_Mem[ele_id];
                            sum += Fea_Value_Mem[reduced_elementsize];
                            ++reduced_elementsize;
                        }
                    }
                    Seg_Idx_Mem[seg_id] = reduced_elementsize;
                    Seg_Len_Mem[seg_id] = sum;
                }                
            }
            elementsize = reduced_elementsize;
            fea_Idx.Size = elementsize;
            fea_Value.Size = elementsize;
        }

        public void Batch_In_GPU()
        {
            sample_Idx.CopyIntoCuda();
            seg_Idx.CopyIntoCuda();
            seg_Margin.CopyIntoCuda();
            seg_Len.CopyIntoCuda();
            fea_Idx.CopyIntoCuda();
            fea_Value.CopyIntoCuda();
        }

        public void Dispose()
        {
            sample_Idx.Dispose();
            seg_Idx.Dispose();
            seg_Margin.Dispose();
            seg_Len.Dispose();
            fea_Idx.Dispose();
            fea_Value.Dispose();
        }
    }
    
}
