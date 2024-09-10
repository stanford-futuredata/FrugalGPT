# 🎓 FrugalGPT: Better Quality and Lower Cost for LLM Applications


The FrugalGPT framework offers a collection of techniques for _building LLM applications with budget constraints_.

## 🚀 Getting Started

You can directly run the  [Google Colab Notebook](https://colab.research.google.com/drive/1LM-Wq-u87VI4TKM4thpnwepnOxTAtWaM?authuser=1#scrollTo=a95a1eec) to experience FrugalGPT. You don't even need API keys to get started with it.

Once you go through the notebook, you'll be ready to build your own LLM applcations with FrugalGPT! 


## 🔧 Installation
You can also install FrugalGPT locally by running the following commands:

```
git clone https://github.com/stanford-futuredata/FrugalGPT
cd FrugalGPT
pip install git+https://github.com/stanford-futuredata/FrugalGPT
wget  https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.zip
unzip HEADLINES.zip -d strategy/
rm HEADLINES.zip
wget -P db/ https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.sqlite
wget -P db/ https://github.com/lchen001/DataHolder/releases/download/v0.0.1/qa_cache.sqlite
```
 

Now you are ready to use the [local intro notebook](intro.ipynb)!



## 📚 Read More


You can get an overview via our Twitter threads:
* [**Introducing**](https://twitter.com/james_y_zou/status/1656285537185980417?cxt=HHwWgoCzqfa6p_wtAAAA)  [**FrugalGPT**](https://twitter.com/matei_zaharia/status/1656295461953650688?cxt=HHwWgIC2zc_8q_wtAAAA) (May 10, 2023)

And read more in the academic paper:
* [**FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance**](https://arxiv.org/pdf/2305.05176.pdf)

A detailed blog with code examples:
* [**Implementing FrugalGPT: Reducing LLM Costs & Improving Performance**](https://portkey.ai/blog/implementing-frugalgpt-smarter-llm-usage-for-lower-costs/)

## 📣 Updates & Changelog

### 🔹 2024.09.10 - Added support to more recent models

- ✅ Added supports of a few new models. This includes proprietary models such as GPT-4o, GPT-4-Turbo, and GPT-4o-mini, and a few open-source models such as Llama 3 (70B) and Gemma 2 (9B).

### 🔹 2024.01.01 - Extracted API generations 

  - ✅ Added the generations from 12 commercial LLM APIs for each dataset evaluated in the paper
  - ✅ Included both input queries and associated parameters (e.g., temperature and stop token)
  - ✅ Released them as CSV files [here](https://github.com/stanford-futuredata/FrugalGPT/releases/tag/0.0.1)
    
## 🎯 Reference

If you use FrugalGPT in a research paper, please cite our work as follows:


```
@article{chen2023frugalgpt,
  title={FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance},
  author={Chen, Lingjiao and Zaharia, Matei and Zou, James},
  journal={arXiv preprint arXiv:2305.05176},
  year={2023}
}
```
