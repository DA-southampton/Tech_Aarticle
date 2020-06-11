# 深度学习模型实战-深度学习模型在各大公司实际生产环境的应用讲解文章
建这个仓库的是因为工作之后发现生产环境中应用的模型需要做到速度和效果的平衡，并不是越复杂越好。所以一味的追求新的模型效果不大（并不是不追，也要多看新东西）。学到模型最终是要用，而且要用好，于是就建了这么个仓库，积累一下深度学习模型在各个公司中的应用以及细节，这样在自己工作中可以做到借鉴。主要是罗列一些各大公司分享的文章，涉及到搜索/推荐/自然语言处理(NLP)，持续更新...

## 部署

在我实际工作中，一般来说部署就是Flask+负载均衡，或者Grpc来提供服务。这个模块积累一下我看到不错的模型部署不错的文章

[蘑菇街自研服务框架如何提升在线推理效率？](https://mp.weixin.qq.com/s/IzLtn1SR-aFuxXM3GNZbFw)
简单介绍：使用协程解决并发问题，使用FLask提供Restful接口，进行容器化部署

## 搜索：

[Albert在房产领域的应用-202005](https://mp.weixin.qq.com/s?src=11&timestamp=1591787166&ver=2392&signature=VnSNP3xcAkOfae88FpCHo-R1DQXzXqtgmrNTELpOoUEUtGHw0EF7HkJt5F4jVKYq-HVuAT1xv1PwtwQLSKhgbYhtx6v0KLf08E*W8xEai6OgvOTT4daSO*2NZY-giofW&new=1)

简单介绍：讲的是贝壳用ALbert做意图分类，和Fasttext相比，提了大概5个点

[DSSM文本匹配模型在苏宁商品语义召回上的应用-201909](https://ai.51cto.com/art/201909/603290.htm)

简单介绍：详细介绍了DSSM模型在苏宁召回的使用，细节很多，居然还点出用的hanlp做的分词（也太细了吧），推荐大家看看

[Transformer在美团搜索排序中的实践-202004](https://tech.meituan.com/2020/04/16/transformer-in-meituan.html)
简单介绍：（引用文中原句）本文旨在分享 Transformer 在美团搜索排序上的实践经验。内容会分为以下三个部分：第一部分对 Transformer 进行简单介绍，第二部分会介绍 Transfomer 在美团搜索排序上的应用以及实践经验，最后一部分是总结与展望。希望能对大家有所帮助和启发。

[深度学习在文本领域的应用-201808](https://tech.meituan.com/2018/06/21/deep-learning-doc.html)

简单介绍：美团的文章，主要是讲了基于深度学习的的文本匹配和排序模型。其中讲了DSSM和变种，引申出来美团自己的ClickNet，基于美团场景进行了优化，大家可以细看一下

[基于BERT，神马搜索在线预测性能如何提升？-201908](https://developer.aliyun.com/article/714552)

简单介绍：讲了一下在神马搜索中bert的性能优化细节，大致就是使用了知乎的cuBert，然后重写了预测逻辑

[阿里文娱搜索算法实践与思考-202006](https://mp.weixin.qq.com/s?src=11&timestamp=1591784596&ver=2392&signature=xsYYdd4UJzPrf6ZzFqnvqJTqf5aaHelBl9-vK9gLMSEDuN9ntXb9ZxM89Zcn*ylB0J-yBOyPUaVKU3QzrTQv8hU4I007NIw2*vbZfvctCrzhzIioU3WSJKuXlnRx*fP0&new=1)
简单介绍：（引用文中原句）本文将以优酷为例，分享视频搜索的算法实践，首先介绍优酷搜索的相关业务和搜索算法体系，从搜索相关性和排序算法的特点和挑战到技术实践方案的落地，最后会深入介绍优酷在多模态视频搜索上的探索和实践。

[视频搜索太难了！阿里文娱多模态搜索算法实践-202005](https://www.infoq.cn/article/16UENbPwYMX7YZC0bhyL)

简单介绍：对比上一个看

[XGBoost在携程搜索排序中的应用-201912](https://mp.weixin.qq.com/s?src=11&timestamp=1591786531&ver=2392&signature=hW8Du7a5sFL*BvkQ8qbnTSUNDfZtYoHL68DKdDFHFPAsb4ndTi9EXlmT-TyPstif0QYq9Z040LlQabdTs9e2UVpmhh5gD3M21BVeN24Y1TSvPBJmKMMRTMBNe6goPYuS&new=1)

简单介绍：如题

[爱奇艺搜索排序模型迭代之路-201909](https://cloud.tencent.com/developer/article/1500313)

简单介绍：如题

[滴滴搜索系统的深度学习演进之路-201908](https://www.infoq.cn/article/90ByjIRA29uxNO0zStsy)

简单介绍：如题

[深度学习在 360 搜索广告 NLP 任务中的应用-201907](https://www.infoq.cn/article/WZR0b9cjkse8uKgKd*eX)
简单介绍：本文作者比对了DSSM 和 ESIM 以及Bert三种模型，介绍了三种模型在实际工作中的应用细节

[小米移动搜索中的 AI 技术-201906](https://www.infoq.cn/article/1pcW2hMQt6wsFxaN*srw)

简单介绍：大概讲了一下搜索中用的技术，比如文本相似度-dssm，具体的看文章吧，没有太多干货

[深度学习在搜索业务中的探索与实践-美团-201901](https://tech.meituan.com/2019/01/10/deep-learning-in-meituan-hotel-search-engine.html)
简单介绍：（引用文中原句）本文会首先介绍一下酒店搜索的业务特点，作为O2O搜索的一种，酒店搜索和传统的搜索排序相比存在很大的不同。第二部分介绍深度学习在酒店搜索NLP中的应用。第三部分会介绍深度排序模型在酒店搜索的演进路线，因为酒店业务的特点和历史原因，美团酒店搜索的模型演进路线可能跟大部分公司都不太一样。最后一部分是总结。

[深度学习在搜狗无线搜索广告中的应用-201803](https://cloud.tencent.com/developer/article/1063013)
简单介绍：（引用文中原句）本次分享主要介绍深度学习在搜狗无线搜索广告中有哪些应用场景，以及分享了我们的一些成果，重点讲解了如何实现基于多模型融合的CTR预估，以及模型效果如何评估，最后和大家探讨DL、CTR 预估的特点及未来的一些方向。




## 推荐：

[双 DNN 排序模型：在线知识蒸馏在爱奇艺推荐的实践-202002](https://www.infoq.cn/article/pUfNBe1o6FwiiPkxQy7C)

简单介绍：还没看。。但是看文章写得效果很厉害，“其中，在爱奇艺短视频场景时长指标 +6.5%，点击率指标 +2.3%；图文推荐场景时长指标 +4.5%，点击率指标 +14% ”（引用自原文）

[美团BERT的探索和实践-201911](https://tech.meituan.com/2019/11/14/nlp-bert-practice.html)

简单介绍：Bert在美团场景中的改进和优化，很厉害，细节很多

[wide&deep 在贝壳推荐场景的实践-201912](https://mp.weixin.qq.com/s?__biz=MzI2ODA3NjcwMw%3D%3D&mid=2247483659&idx=1&sn=deb9c5e22eabd3c52d2418150a40c68a&scene=45#wechat_redirect)

简单介绍：如题，看了之后感觉还不错，

[贝壳找房的深度学习模型迭代及算法优化-201910](https://cloud.tencent.com/developer/article/1528388)

简单介绍：（引用文中原句）

第一阶段：建立初版模型系统，采用 XGBoost 模型，完成项目从 0 到 1 的过程；
第二阶段：深度学习模型，采用 DNN+RNN 混合模型；
第三阶段：效果持续优化，也是业务需要。

[千人千面营销系统在携程金融支付的实践](https://cloud.tencent.com/developer/article/1500371)

简单介绍：（引自原文）支付中心数据组开发的一套用户精准营销系统

[爱奇艺个性化推荐排序实践-201907](http://www.iqiyi.com/common/20171025/46d31f38d4cb7ee2.html)

简单介绍：还没看

[强化学习在携程酒店推荐排序中的应用探索](https://cloud.tencent.com/developer/article/1449819)

简单介绍：（引用原文）我们尝试在城市欢迎度排序场景中引入了强化学习。通过实验发现，增加强化学习后，能够在一定程度上提高排序的质量。

[搜狗信息流推荐算法实践-201904](https://www.infoq.cn/article/A9w0Xg-P1vqbUZ4cEmyH)

简单介绍：还没看。。。

[深度学习在美团点评推荐业务中实践-201901](https://zhuanlan.zhihu.com/p/55023302)
简单介绍：（引用文中原句）在推荐平台的构建过程中，多策略选品和排序是两个非常重要的部分，本文接下来主要介绍深度学习相关的推荐算法，主要包括 DSSM、Session Based RNN 推荐召回模型与 Wide Deep Learning 的排序模型，我们会介绍深度学习模型在推荐业务应用及实现的相关细节，包括模型原理、线上效果、实践经验及思考。

[美团“猜你喜欢”深度学习排序模型实践-201803](https://tech.meituan.com/2018/03/29/recommend-dnn.html)
简单介绍：（引用文中原句）目前，深度学习模型凭借其强大的表达能力和灵活的网络结构在诸多领域取得了重大突破，美团平台拥有海量的用户与商家数据，以及丰富的产品使用场景，也为深度学习的应用提供了必要的条件。本文将主要介绍深度学习模型在美团平台推荐排序场景下的应用和探索。

[优酷视频基于用户兴趣个性化推荐的挑战和实践-201802](https://developer.aliyun.com/article/443621?scm=20140722.184.2.173)
简单介绍：（引用文中原句）本文将介绍一下优酷个性化搜索推荐的服务，优酷在视频个性化搜索推荐里用户兴趣个性化表达碰到的挑战和问题，当前工业界常用的方法，以及我们针对这些问题的尝试。

[深度学习在58同城智能推荐系统中的应用实践-201802](https://mp.weixin.qq.com/s/qCpCHueEK7Nja-cPmlCaCg?)

简单介绍：还没看

[携程个性化推荐算法实践-201801](https://zhuanlan.zhihu.com/p/32785759)

[深度学习在美团点评推荐平台排序中的运用-201707](https://tech.meituan.com/2017/07/28/dl.html)



## 智能客服：

[语义匹配在贝壳找房智能客服中的应用-202005](https://mp.weixin.qq.com/s?src=11&timestamp=1591783120&ver=2392&signature=RZJ5qcZ5PEc0eHDi9eznGXdaoQM2s2WEgsQgMlft5aPuOUiveyUcsoMCIm-sefmm8sRV2OpzrpsoaR6xAv8He0Q84azUJ5wv5gcvB1KQcx7OyN7A1b0QIt2xIpvhSSRH&new=1)

简单介绍：还没看
