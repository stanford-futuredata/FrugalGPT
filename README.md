# 🎓 FrugalGPT: Better Quality and Lower Cost for LLM Applications


The FrugalGPT framework offers a collection of techniques for _building LLM applications with budget constraints_.

## 🔧 Installation
To install FrugalGPT, run the following commands:

`git clone https://github.com/stanford-futuredata/FrugalGPT`

`cd FrugalGPT`

`pip install git+https://github.com/stanford-futuredata/FrugalGPT`

`wget  https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.zip`

` unzip HEADLINES.zip -d strategy/`

`rm HEADLINES.zip`

`wget -P db/ https://github.com/lchen001/DataHolder/releases/download/v0.0.1/HEADLINES.sqlite`

`wget -P db/ https://github.com/lchen001/DataHolder/releases/download/v0.0.1/qa_cache.sqlite`
 




Now you are ready to use the [intro notebook](intro.ipynb)!

## 🚀 Getting Started

Our [intro notebook](intro.ipynb) provides examples of using FrugalGPT in a few different applications.

You can also directly start with the  [Google Colab Notebook](https://colab.research.google.com/drive/1LM-Wq-u87VI4TKM4thpnwepnOxTAtWaM?authuser=1#scrollTo=a95a1eec). You don't even need API keys to get started with it.

Once you go through the notebook, you'll be ready to build your own LLM applcations with FrugalGPT! 


## 📚 Read More


You can get an overview via our Twitter threads:
* [**Introducing**](https://twitter.com/james_y_zou/status/1656285537185980417?cxt=HHwWgoCzqfa6p_wtAAAA)  [**FrugalGPT**](https://twitter.com/matei_zaharia/status/1656295461953650688?cxt=HHwWgIC2zc_8q_wtAAAA) (May 10, 2023)

And read more in the academic paper:
* [**FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance**](https://arxiv.org/pdf/2305.05176.pdf)

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