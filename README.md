<div align="center">

    MoFE-Time: Mixture of Frequency Domain Experts for Time-Series Forecasting Models
 </b></h2>
</div>

<div align="center">


</div>

<div align="center">



</div>

## Abstract
Time series forecasting is a fundamental task with broad applications across various domains. Recently, inspired by the success of large language models (LLMs), foundation models for time series gained significant attention. However, most existing approaches directly adopt vanilla transformers, which underexplores the joint modeling of temporal and frequency characteristics, resulting in limited performance on complex time series. To address this, we propose MoFE-Time, a novel time series forecasting foundation model that integrates temporal and frequency-domain representations within a Mixture of Experts (MoE) framework. Specifically, we design Frequency and Time Cells (FTC) as experts following attention modules, and employ an MoE routing mechanism to construct multidimensional sparse representations of input signals. Extensive experiments on six public benchmarks demonstrate that MoFE-Time achieves new state-of-the-art results. Furthermore, we construct a proprietary real-world dataset, NEV-sales, to evaluate the model's practical effectiveness. MoFE-Time consistently outperforms competitive baselines on this dataset, demonstrating its potential for real-world commercial applications.



## Usage
### pretrain
1. data prepare

   download from huggingface
   https://huggingface.co/datasets/Maple728/Time-300B
3. Install Pytorch and other dependencies.
   ```
   pip install -r requirements.txt
   ```
4. start pretrain
   
    ```sh ./src/pretrain_and_eval_ds.sh```
   
   Pretrain on Multiple Nodes
    ```sh ./src/pretrain_and_eval_nodes.sh```
### fine tune

    ```sh ./src/fine_tune_ds.sh```


