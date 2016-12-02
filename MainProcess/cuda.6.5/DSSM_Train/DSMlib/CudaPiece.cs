using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DSMlib
{
    
    /// <summary>
    /// xinson, 3/20/14. A cuda piece serves as a bridge/connect of the same piece of data between CPU and GPU.
    /// </summary>
    public class CudaPieceFloat:IDisposable
    {
        int size = 0;
        /// <summary>
        /// Enable user to set the actual effective size of data
        /// </summary>
        public int Size { get { return size; }
            set
            {
                if (value > cpuMemArray.Length)
                {
                    throw new Exception("CudaPieceFloat set length cannot be greater than the allocated buffer!");
                }
                size = value;
            }
        }
        float[] cpuMemArray = null;

        public float[] MemPtr
        {
            get { return cpuMemArray; }
        }
        IntPtr cudaPiecePointer = IntPtr.Zero;

        public IntPtr CudaPtr
        {
            get { return cudaPiecePointer; }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="length"></param>
        /// <param name="needCpuMem"></param>
        public CudaPieceFloat(int length, bool needCpuMem, bool needGpuMem)         
        {
            // the input is given assuming MATH_LIB = gpu
            // So if cpu is used, we will overwrite
            if (ParameterSetting.MATH_LIB == MathLibType.cpu)
            {
                needCpuMem = true;
                needGpuMem = false;
            }
            size = length;
            if (needCpuMem)
            {
                cpuMemArray = new float[size];
            }
            if (needGpuMem)
            {
                if ((Int64)(cudaPiecePointer = Cudalib.CudaAllocFloat(size)) == 0)
                {
                    throw new Exception("Out of GPU Memo, use a smaller model!");
                }
            }
        }
        ~CudaPieceFloat()
        {
            Dispose();
        }

        /// <summary>
        /// Copy data from GPU to CPU
        /// </summary>
        unsafe public void CopyOutFromCuda()
        {
            if (cudaPiecePointer == IntPtr.Zero)
            {
                return;
            }
            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for CopyOutFromCuda() operation!");
            }
            fixed (float* pb = &cpuMemArray[0])
            {
                Cudalib.CudaCopyOutFloat(cudaPiecePointer, (IntPtr)pb, Size);
            }
        }

        /// <summary>
        /// Copy data from GPU to CPU
        /// </summary>
        unsafe public void CopyOutFromCuda(int SpecifiedSize)
        {
            if (cudaPiecePointer == IntPtr.Zero)
            {
                return;
            }
            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for CopyOutFromCuda() operation!");
            }
            fixed (float* pb = &cpuMemArray[0])
            {
                Cudalib.CudaCopyOutFloat(cudaPiecePointer, (IntPtr)pb, SpecifiedSize);
            }
        }

        /// <summary>
        /// Copy data from CPU to GPU
        /// </summary>
        unsafe public void CopyIntoCuda()
        {
            if (cudaPiecePointer == IntPtr.Zero)
            {
                return;
            }
            
            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for CopyIntoCuda() operation!");
            }
            fixed (float* pb = &cpuMemArray[0])
            {
                Cudalib.CudaCopyInFloat(cudaPiecePointer, (IntPtr)pb, Size);
            }
        }

        /// <summary>
        /// Copy data from CPU to GPU
        /// </summary>
        unsafe public void CopyIntoCuda(int SpecifiedSize)
        {
            if (cudaPiecePointer == IntPtr.Zero)
            {
                return;
            }

            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for CopyIntoCuda() operation!");
            }
            fixed (float* pb = &cpuMemArray[0])
            {
                Cudalib.CudaCopyInFloat(cudaPiecePointer, (IntPtr)pb, SpecifiedSize);
            }
        }

        public void Init(float scale, float bias)
        {
            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for Init() operation!");
            }
            Random random = ParameterSetting.Random;
            for (int i = 0; i < Size; ++i)
            {
                cpuMemArray[i] = (float) ( random.NextDouble() * scale + bias );
            }
            CopyIntoCuda();
        }
        public void Init(float value)
        {
            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for Init() operation!");
            }
            for (int i = 0; i < Size; ++i)
            {
                cpuMemArray[i] = value;
            }            
            CopyIntoCuda();
        }
        public void Init(float[] data)
        {
            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for Init() operation!");
            }
            if (size != data.Length)
            {
                throw new Exception("Error! Init(float[]). Input float array has different size than expected!");
            }
            data.CopyTo(cpuMemArray, 0);            
            CopyIntoCuda();
        }

        /// <summary>
        /// 
        /// </summary>
        public void Zero()
        {
            if (cpuMemArray != null)
            {
                for (int i = 0; i < Size; ++i)
                {
                    cpuMemArray[i] = 0;
                }
            }
            if (cudaPiecePointer != IntPtr.Zero)
            {
                Cudalib.Zero(cudaPiecePointer, Size);
            }
        }

        public void Dispose()
        {
            if (cudaPiecePointer != IntPtr.Zero)
            {
                Cudalib.CudaDeallocFloat(cudaPiecePointer);
                cudaPiecePointer = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// xinson, 3/20/14. A cuda piece serves as a bridge/connect of the same piece of data between CPU and GPU.
    /// </summary>
    public class CudaPieceInt:IDisposable
    {
        int size;
        /// <summary>
        /// Enable user to set the actual effective size of data
        /// </summary>
        public int Size
        {
            get { return size; }
            set
            {
                if (value > cpuMemArray.Length)
                {
                    throw new Exception("CudaPieceFloat set length cannot be greater than the allocated buffer!");
                }
                size = value;
            }
        }
        int[] cpuMemArray = null;
        public int[] MemPtr
        {
            get { return cpuMemArray; }
        }
        IntPtr cudaPiecePointer = IntPtr.Zero;

        public IntPtr CudaPtr
        {
            get { return cudaPiecePointer; }
        }
        
        public CudaPieceInt(int length, bool needCpuMem, bool needGpuMem)         
        {
            // the input is given assuming MATH_LIB = gpu
            // So if cpu is used, we will overwrite
            if (ParameterSetting.MATH_LIB == MathLibType.cpu)
            {
                needCpuMem = true;
                needGpuMem = false;
            }
            size = length;
            if (needCpuMem)
            {
                cpuMemArray = new int[size];
            }
            if (needGpuMem)
            {
                if ((Int64)(cudaPiecePointer = Cudalib.CudaAllocInt(size)) == 0)
                {
                    throw new Exception("Out of GPU Memo, use a smaller model!");
                }
            }
        }

        ~CudaPieceInt()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (cudaPiecePointer != IntPtr.Zero)
            {
                Cudalib.CudaDeallocInt(cudaPiecePointer);
                cudaPiecePointer = IntPtr.Zero;
            }
        }
        unsafe public void CopyIntoCuda()
        {
            if (cudaPiecePointer == IntPtr.Zero)
            {
                return;
            }

            if (cpuMemArray == null)
            {
                throw new Exception("Error! Must set needCpuMem=true for CopyIntoCuda() operation!");
            }
            fixed (int* gpu_ptr = cpuMemArray)
            {
                Cudalib.CudaCopyInInt(cudaPiecePointer, (IntPtr)gpu_ptr, size);
                //Amplib.CopyInInt(GPU_negative_index[i], (IntPtr)gpu_neg, batchSize);
            }
        }

    }

    
}
