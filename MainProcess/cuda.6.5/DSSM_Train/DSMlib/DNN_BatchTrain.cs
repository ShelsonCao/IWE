using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DSMlib
{
    /// <summary>
    /// Train DNN with Batch Learning.
    /// </summary>
    public class DNN_BatchTrain
    {
        public int Num { get; set; }
        public float LearnRate { get { return LearningParameters.learning_rate; } set { LearningParameters.learning_rate = value; } }
        public DNN_BatchTrain(DNN dnn)
        {
            Num = dnn.ModelParameterNumber;            
        }
        
        ~DNN_BatchTrain()
        {}

        /// <summary>
        /// Initalize Batch Grad.
        /// </summary>
        public virtual void StartBatch()
        { }

        public virtual void AggragateBatch(DNNRun dnnData)
        { }

        /// <summary>
        /// End of Batch Grad.
        /// </summary>
        public virtual void EndBatch()
        { }

        public virtual void Init(DNN dnn)
        { }

        public virtual void Update(DNN dnn)
        { }

        
    }

    /// <summary>
    /// βHS(k) = gT(k+1)y(k) / dT(k) y(k)  (1952) in the original (linear) CG paper of Hestenes and Stiefel [59]
    /// yk = gk+1−gk
    /// dk+1 =−gk+1 + βk dk, d0 =−g0
    /// xk+1 = xk + αk dk
    /// </summary>
    public class DNN_BatchTrain_CG_HS : DNN_BatchTrain
    {
        CudaPieceFloat parameters = null;
        List<CudaPieceFloat> grad_list = new List<CudaPieceFloat>();
        CudaPieceFloat direction = null;

        public int GradHistory { get { return 2; } }
        int GradIdx { get; set; }
        
        
        public DNN_BatchTrain_CG_HS(DNN dnn)
            : base(dnn)
        {
            parameters = new CudaPieceFloat(Num, true, true);
            direction = new CudaPieceFloat(Num, true, true);
            for (int i = 0; i < GradHistory; i++)
            {
                grad_list.Add(new CudaPieceFloat(Num, true, true));
            }
        }

        /// <summary>
        /// Init Model Parameters.
        /// </summary>
        /// <param name="dnnData"></param>
        public override void Init(DNN dnn)
        {
            MathOperatorManager.GlobalInstance.Zero(parameters, Num);
            int ParNum = 0;
            for (int i = 0; i < dnn.neurallinks.Count; i++)
            {
                dnn.neurallinks[i].weight.CopyOutFromCuda();

                int mnum = dnn.neurallinks[i].Neural_In.Number * dnn.neurallinks[i].N_Winsize * dnn.neurallinks[i].Neural_Out.Number;
                MathOperatorManager.GlobalInstance.Matrix_Add_OFFSET(parameters, ParNum, dnn.neurallinks[i].weight, 0, mnum, 1.0f);
                ParNum += mnum;

                if (ParameterSetting.UpdateBias)
                {
                    mnum = dnn.neurallinks[i].Neural_Out.Number;
                    MathOperatorManager.GlobalInstance.Matrix_Add_OFFSET(parameters, ParNum, dnn.neurallinks[i].bias, 0, mnum, 1.0f);
                    ParNum += mnum;
                }
            }

            parameters.CopyOutFromCuda();
            GradIdx = 0;
        }

        public override void Update(DNN dnn)
        {
            int ParNum = 0;
            for (int i = 0; i < dnn.neurallinks.Count; i++)
            {
                int mnum = dnn.neurallinks[i].Neural_In.Number * dnn.neurallinks[i].N_Winsize * dnn.neurallinks[i].Neural_Out.Number;
                MathOperatorManager.GlobalInstance.Zero(dnn.neurallinks[i].weight, mnum);
                MathOperatorManager.GlobalInstance.Matrix_Add_OFFSET(dnn.neurallinks[i].weight, 0, parameters, ParNum, mnum, 1.0f);
                ParNum += mnum;

                if (ParameterSetting.UpdateBias)
                {
                    mnum = dnn.neurallinks[i].Neural_Out.Number;
                    MathOperatorManager.GlobalInstance.Zero(dnn.neurallinks[i].bias, mnum);
                    MathOperatorManager.GlobalInstance.Matrix_Add_OFFSET(dnn.neurallinks[i].bias, 0, parameters, ParNum, mnum, 1.0f);
                    ParNum += mnum;
                }
            }
            GradIdx += 1;
        }

        /// <summary>
        /// Initalize Batch Grad.
        /// </summary>
        public override void StartBatch()
        {
            MathOperatorManager.GlobalInstance.Zero(grad_list[GradIdx % GradHistory], Num);
        }

        public override void AggragateBatch(DNNRun dnnData)
        {
            int ParNum = 0;
            for (int i = 0; i < dnnData.neurallinks.Count; i++)
            {
                int mnum = dnnData.neurallinks[i].NeuralLinkModel.Neural_In.Number * dnnData.neurallinks[i].NeuralLinkModel.N_Winsize * dnnData.neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                MathOperatorManager.GlobalInstance.Matrix_Add_OFFSET(grad_list[GradIdx % GradHistory], ParNum, dnnData.neurallinks[i].WeightDeriv, 0, mnum, -1.0f);
                ParNum += mnum;

                if (ParameterSetting.UpdateBias)
                {
                    mnum = dnnData.neurallinks[i].NeuralLinkModel.Neural_Out.Number;
                    MathOperatorManager.GlobalInstance.Matrix_Add_OFFSET(grad_list[GradIdx % GradHistory], ParNum, dnnData.neurallinks[i].BiasDeriv, 0, mnum, -1.0f);
                    ParNum += mnum;
                }
            }
        }

        /// <summary>
        /// End of Batch Grad.
        /// </summary>
        public override void EndBatch()
        {
            if (GradIdx == 0)
            {
                MathOperatorManager.GlobalInstance.Zero(direction, Num);
                MathOperatorManager.GlobalInstance.Matrix_Add_OFFSET(direction, 0, grad_list[GradIdx % GradHistory], 0, Num, 1);
            }
            else
            {
                //yk = gk+1−gk   -yk = gk - gk+1
                //βHS(k) = gT(k+1)y(k) / dT(k) y(k)  (1952) in the original (linear) CG paper of Hestenes and Stiefel [59]
                
                //Cudalib.Matrix_Add_OFFSET(grad_list[(GradIdx - 1) % GradHistory].CudaPtr, 0, grad_list[GradIdx % GradHistory].CudaPtr, 0, Num, -1);
                /*grad_list[(GradIdx - 1) % GradHistory].CopyOutFromCuda();
                direction.CopyOutFromCuda();
                grad_list[GradIdx % GradHistory].CopyOutFromCuda();

                float g1 = BasicMathlib.VectorInnerProduct(grad_list[GradIdx % GradHistory].MemPtr, 0, grad_list[GradIdx % GradHistory].MemPtr, 0, Num);
                float g2 = BasicMathlib.VectorInnerProduct(grad_list[GradIdx % GradHistory].MemPtr, 0, grad_list[(GradIdx - 1) % GradHistory].MemPtr, 0, Num);
                float s1 = BasicMathlib.VectorInnerProduct(direction.MemPtr, 0, grad_list[(GradIdx ) % GradHistory].MemPtr, 0, Num);
                float s2 = BasicMathlib.VectorInnerProduct(direction.MemPtr, 0, grad_list[(GradIdx - 1) % GradHistory].MemPtr, 0, Num);
                //float dy = BasicMathlib.VectorInnerProduct(direction.MemPtr, 0, grad_list[(GradIdx - 1) % GradHistory].MemPtr, 0, Num);

                float beta = 0;
                if (Math.Abs(s1-s2) > float.Epsilon)
                {
                    beta = (g1 - g2) / (s1 - s2);
                }
                */
                
                grad_list[(GradIdx - 1) % GradHistory].CopyOutFromCuda();
                grad_list[GradIdx % GradHistory].CopyOutFromCuda();
                float gk = BasicMathlib.VectorInnerProduct(grad_list[GradIdx % GradHistory].MemPtr, 0, grad_list[GradIdx % GradHistory].MemPtr, 0, Num);
                float gk_1 = BasicMathlib.VectorInnerProduct(grad_list[(GradIdx - 1) % GradHistory].MemPtr, 0, grad_list[(GradIdx - 1) % GradHistory].MemPtr, 0, Num);
                float beta = gk * 1.0f / (gk_1 + float.Epsilon);
                
                Console.WriteLine("Beta Value ....................." + beta.ToString());
                //dk+1 =−gk+1 + βk dk, d0 =−g0
                MathOperatorManager.GlobalInstance.Scale_Matrix(direction, 1, Num, beta); // grad_list[GradIdx % GradHistory].CudaPtr, 0, Num, 1);
                MathOperatorManager.GlobalInstance.Matrix_Add(direction, grad_list[GradIdx % GradHistory], 1, Num, 1.0f);
            }

            //xk+1 = xk + αk dk
            MathOperatorManager.GlobalInstance.Matrix_Add(parameters, direction, 1, Num, -LearnRate);
        }

        ~DNN_BatchTrain_CG_HS()
        {
            if (parameters != null)
            {
                parameters.Dispose();
            }

            if (direction != null)
            {
                direction.Dispose();
            }

            for (int i = 0; i < grad_list.Count; i++)
            {
                grad_list[i].Dispose();
            }
        }
    }
}
