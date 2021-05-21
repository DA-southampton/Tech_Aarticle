## 实战深度学习模型文章阅读积累，主要是搜广推，欢迎关注；

微信公众号： NLP从入门到放弃

![微信公众号](./images/wechat.png)


建这个仓库的是因为工作之后发现生产环境中应用的模型需要做到速度和效果的平衡，并不是越复杂越好。所以一味的追求新的模型效果不大（并不是不追，也要多看新东西）。学到模型最终是要用，而且要用好，于是就建了这么个仓库，积累一下深度学习模型在各个公司中的应用以及细节，这样在自己工作中可以做到借鉴。主要是罗列一些各大公司分享的文章，涉及到搜索/推荐/自然语言处理(NLP)，持续更新...

1. 模型部署
2. 搜索
3. 推荐
4. 多模态
5. NLP
6. 推荐系统

## 1. 部署

在我实际工作中，一般来说部署就是Flask+负载均衡，或者Grpc来提供服务。这个模块积累一下我看到不错的模型部署不错的文章

[蘑菇街自研服务框架如何提升在线推理效率？](https://mp.weixin.qq.com/s/IzLtn1SR-aFuxXM3GNZbFw)
使用协程解决并发问题，使用FLask提供Restful接口，进行容器化部署 
[如何解决推荐系统工程难题——深度学习推荐模型线上serving？](https://zhuanlan.zhihu.com/p/77664408)
介绍了几种serving方式，值得一看  
[爱奇艺基于CPU的深度学习推理服务优化实践-201904](https://zhuanlan.zhihu.com/p/61853955) 
 爱奇艺主要是在算法，应用以及系统三个方面对模型的部署进行优化。系统级主要是针对硬件平台上做的一些性能优化的方法，应用级是跟特定应用相关的分析以及优化的方法，算法级是针对算法的优化，例如模型的裁剪，模型的量化，在四个任务上提升了10倍左右（引自原文） 

## 2 搜索：

### 2.1 Query理解：

1. 搜索中的 Query 理解及应用
   https://mp.weixin.qq.com/s/rZMtsbMuyGwcy2KU7mzZhQ

2. 搜索中的Query扩展技术
   https://mp.weixin.qq.com/s/WRVwKaWvY-j-bkjxCprckQ

3. 再谈搜索中的Query扩展技术
   https://mp.weixin.qq.com/s/q4aPtUYi27h-0sqD4bokQQ

4. 搜索query意图识别的演进-微信AI
   https://mp.weixin.qq.com/s/0Hh_iV8tNFd0eEpXSxy9nA

5. [Albert在房产领域的应用-意图识别](https://github.com/DA-southampton/Tech_Aarticle/blob/master/content/精读-ALBERT在房产领域的实践.md) 

讲的是贝壳用ALbert做意图分类，和Fasttext相比，提了大概8个点，推理速度耗时20ms 

6. [达观数据搜索引擎的Query自动纠错技术和架构详解](http://t.cn/Rql7mz9) 
7.  [阿里文娱搜索算法实践与思考](https://www.infoq.cn/article/RUlwIBXPmUKILgqiyR4I) 

### 2.2 搜索相关性排序

1. [Transformer在美团搜索排序中的实践-202004](https://tech.meituan.com/2020/04/16/transformer-in-meituan.html)

（引用文中原句）本文旨在分享 Transformer 在美团搜索排序上的实践经验。内容会分为以下三个部分：第一部分对 Transformer 进行简单介绍，第二部分会介绍 Transfomer 在美团搜索排序上的应用以及实践经验，最后一部分是总结与展望。希望能对大家有所帮助和启发。

2. [深度学习在文本领域的应用-201808](https://tech.meituan.com/2018/06/21/deep-learning-doc.html)  美团的文章，主要是讲了基于深度学习的的文本匹配和排序模型。其中讲了DSSM和变种，引申出来美团自己的ClickNet，基于美团场景进行了优化，大家可以细看一下 

3. 京东电商搜索中的语义检索与商品排序
   https://mp.weixin.qq.com/s/4UBehc0eikVqcsFP7xL_Zw

4. 阿里文娱深度语义搜索相关性探索
   https://mp.weixin.qq.com/s/1aNd3dxwjCKUJACSq1uF-Q

5. 机器学习在高德搜索建议中的应用优化实践
   https://mp.weixin.qq.com/s/D3qxlzZgwnMprzEVuMpmgg

6. [DSSM文本匹配模型在苏宁商品语义召回上的应用-201909](https://ai.51cto.com/art/201909/603290.htm) 

详细介绍了DSSM模型在苏宁召回的使用，细节很多，居然还点出用的hanlp做的分词（也太细了吧），推荐大看看 

7.  [基于BERT，神马搜索在线预测性能如何提升？-201908](https://developer.aliyun.com/article/714552)  讲了一下在神马搜索中bert的性能优化细节，大致就是使用了知乎的cuBert，然后重写了预测逻辑 

8. [阿里文娱搜索算法实践与思考-202006](https://mp.weixin.qq.com/s?src=11&timestamp=1591784596&ver=2392&signature=xsYYdd4UJzPrf6ZzFqnvqJTqf5aaHelBl9-vK9gLMSEDuN9ntXb9ZxM89Zcn*ylB0J-yBOyPUaVKU3QzrTQv8hU4I007NIw2*vbZfvctCrzhzIioU3WSJKuXlnRx*fP0&new=1)  （引用文中原句）本文将以优酷为例，分享视频搜索的算法实践，首先介绍优酷搜索的相关业务和搜索算法体系，从搜索相关性和排序算法的特点和挑战到技术实践方案的落地，最后会深入介绍优酷在多模态视频搜索上的探索和实践。

9.  [视频搜索太难了！阿里文娱多模态搜索算法实践-202005](https://www.infoq.cn/article/16UENbPwYMX7YZC0bhyL)  对比上一个看    

10.  [XGBoost在携程搜索排序中的应用-201912](https://mp.weixin.qq.com/s?src=11&timestamp=1591786531&ver=2392&signature=hW8Du7a5sFL*BvkQ8qbnTSUNDfZtYoHL68DKdDFHFPAsb4ndTi9EXlmT-TyPstif0QYq9Z040LlQabdTs9e2UVpmhh5gD3M21BVeN24Y1TSvPBJmKMMRTMBNe6goPYuS&new=1)  如题            

11. [爱奇艺搜索排序模型迭代之路-201909](https://cloud.tencent.com/developer/article/1500313)  如题         

12. [滴滴搜索系统的深度学习演进之路-201908](https://www.infoq.cn/article/90ByjIRA29uxNO0zStsy)  如题           

13. [深度学习在 360 搜索广告 NLP 任务中的应用-201907](https://www.infoq.cn/article/WZR0b9cjkse8uKgKd*eX)  

本文作者比对了DSSM 和 ESIM 以及Bert三种模型，介绍了三种模型在实际工作中的应用细节 

14. [小米移动搜索中的 AI 技术-201906](https://www.infoq.cn/article/1pcW2hMQt6wsFxaN*srw)  

大概讲了一下搜索中用的技术，比如文本相似度-dssm，具体的看文章吧 

15. [深度学习在搜索业务中的探索与实践-美团-201901](https://tech.meituan.com/2019/01/10/deep-learning-in-meituan-hotel-search-engine.html)  

（引用文中原句）本文会首先介绍一下酒店搜索的业务特点，作为O2O搜索的一种，酒店搜索和传统的搜索排序相比存在很大的不同。第二部分介绍深度学习在酒店搜索NLP中的应用。第三部分会介绍深度排序模型在酒店搜索的演进路线，因为酒店业务的特点和历史原因，美团酒店搜索的模型演进路线可能跟大部分公司都不太一样。最后一部分是总结。 

16. [深度学习在搜狗无线搜索广告中的应用-201803](https://cloud.tencent.com/developer/article/1063013)  

（引用文中原句）本次分享主要介绍深度学习在搜狗无线搜索广告中有哪些应用场景，以及分享了我们的一些成果，重点讲解了如何实现基于多模型融合的CTR预估，以及模型效果如何评估，最后和大家探讨DL、CTR 预估的特点及未来的一些方向。 

17. [搜索相关性算法在 DiDi Food 中的探索](https://www.infoq.cn/article/01O8GTA66sakZOtbyUcL)        

### 2.3 其他：

1. 阿里文娱算法公开课#04：算法工程师的核心技能-搜索推荐篇
   https://mp.weixin.qq.com/s/vgrWwSZLbl5svAcrxNuJpg

2.  [说一说视频搜索](https://zhuanlan.zhihu.com/p/144359114?utm_source=wechat_session&utm_medium=social&utm_oi=691775466138251264&utm_content=sec) 

## 3. 推荐

###  3.1 总体介绍

1. 美图个性化推送的 AI 探索之路
   https://mp.weixin.qq.com/s/HRGk5bfaOdj-6X4opEYA-w
2. 信息流推荐的用户增长机制
   https://mp.weixin.qq.com/s/hjeS_nEsvxu0D_Bj2vJe7w
3. 智能推荐算法在花椒直播中的应用
   https://mp.weixin.qq.com/s/ec88cMR4K6pWyHhJs7FEFQ
4.  [智能推荐算法在直播场景中的应用--花椒推荐系统](https://mp.weixin.qq.com/s/fUdKIqygxqlkuv0P4wiIRg) 
5. 58同镇下沉市场中的推荐技术实践
   https://mp.weixin.qq.com/s/j6FWqkdbOdQk-qAmYNmJqQ
6. 信息流推荐在凤凰新闻的业务实践
   https://mp.weixin.qq.com/s/aCTP4OCGyWxWGrlCFHSYJQ
7. 广告算法在阿里文娱用户增长中的实践
   https://mp.weixin.qq.com/s/NVQPv5ua9kxw1MK8UVQcuQ
8. Attention机制在深度学习推荐算法中的应用
   https://mp.weixin.qq.com/s/1LYyiDJBDKVgNjc7a1Qc4A
9. 跨域推荐技术在58部落内容社区的实践
   https://mp.weixin.qq.com/s/YylA34cBEshzb9sFY0gklw
10.  [携程个性化推荐算法实践-201801](https://zhuanlan.zhihu.com/p/32785759) 
11.  [深度学习在58同城智能推荐系统中的应用实践-201802](https://mp.weixin.qq.com/s/qCpCHueEK7Nja-cPmlCaCg?) 
12.  [贝壳找房的深度学习模型迭代及算法优化-201910](https://cloud.tencent.com/developer/article/1528388)  （引用文中原句）第一阶段：建立初版模型系统，采用 XGBoost 模型，完成项目从 0 到 1 的过程； 第二阶段：深度学习模型，采用 DNN+RNN 混合模型； 第三阶段：效果持续优化，也是业务需要。 
13.  [wide&deep 在贝壳推荐场景的实践-201912](https://mp.weixin.qq.com/s?__biz=MzI2ODA3NjcwMw%3D%3D&mid=2247483659&idx=1&sn=deb9c5e22eabd3c52d2418150a40c68a&scene=45#wechat_redirect)  如题，看了之后感觉还不错，         
14. [优酷视频基于用户兴趣个性化推荐的挑战和实践-201802](https://developer.aliyun.com/article/443621?scm=20140722.184.2.173)  简单介绍：（引用文中原句）本文将介绍一下优酷个性化搜索推荐的服务，优酷在视频个性化搜索推荐里用户兴趣个性化表达碰到的挑战和问题，当前工业界常用的方法，以及我们针对这些问题的尝试。 
15.  [深度学习在美团点评推荐业务中实践-201901](https://zhuanlan.zhihu.com/p/55023302)  （引用文中原句）在推荐平台的构建过程中，多策略选品和排序是两个非常重要的部分，本文接下来主要介绍深度学习相关的推荐算法，主要包括 DSSM、Session Based RNN 推荐召回模型与 Wide Deep Learning 的排序模型，我们会介绍深度学习模型在推荐业务应用及实现的相关细节，包括模型原理、线上效果、实践经验及思考。 
16.  [千人千面营销系统在携程金融支付的实践](https://cloud.tencent.com/developer/article/1500371)  （引自原文）支付中心数据组开发的一套用户精准营销系统      
17. [搜狗信息流推荐算法实践-201904](https://www.infoq.cn/article/A9w0Xg-P1vqbUZ4cEmyH)  还没看。。。       
18.  [强化学习在携程酒店推荐排序中的应用探索](https://cloud.tencent.com/developer/article/1449819)  （引用原文）我们尝试在城市欢迎度排序场景中引入了强化学习。通过实验发现，增加强化学习后，能够在一定程度上提高排序的质量。    
19. [双 DNN 排序模型：在线知识蒸馏在爱奇艺推荐的实践-202002](https://www.infoq.cn/article/pUfNBe1o6FwiiPkxQy7C)  还没看。。但是看文章写得效果很厉害，“其中，在爱奇艺短视频场景时长指标 +6.5%，点击率指标 +2.3%；图文推荐场景时长指标 +4.5%，点击率指标 +14% ”（引用自原文） 

### 3.2 召回

1. "全能选手"召回表征算法实践--网易严选
   https://mp.weixin.qq.com/s/s4tNPWQrisYIiMuNUzEtNQ
2. 深度召回在招聘推荐中的挑战和实践
   https://mp.weixin.qq.com/s/mcETNOICbabRRq9BBdL4zw
3. 阿里飞猪个性化推荐：召回篇
   https://mp.weixin.qq.com/s/323D5MFivtrmo3ISwQbpYg

### 3.3 排序

1. 深度学习在商业排序的应用实践-58同城
   https://mp.weixin.qq.com/s/2SRGdFZ9RVl4ljBh5MIUqQ
2. 汽车之家推荐系统排序算法迭代之路-视频
   https://mp.weixin.qq.com/s/3wAR3evFAeKfsCYJ6WLTHQ
3.  [深度学习在美团点评推荐平台排序中的运用-201707](https://tech.meituan.com/2017/07/28/dl.html) 
4.  [美团“猜你喜欢”深度学习排序模型实践-201803](https://tech.meituan.com/2018/03/29/recommend-dnn.html)  （引用文中原句）目前，深度学习模型凭借其强大的表达能力和灵活的网络结构在诸多领域取得了重大突破，美团平台拥有海量的用户与商家数据，以及丰富的产品使用场景，也为深度学习的应用提供了必要的条件。本文将主要介绍深度学习模型在美团平台推荐排序场景下的应用和探索。 
5.  [爱奇艺个性化推荐排序实践-201907](http://www.iqiyi.com/common/20171025/46d31f38d4cb7ee2.html)  还没看              

### 3.4 Embedding

1. 推荐系统 Embedding 技术实践总结
   https://mp.weixin.qq.com/s/7DXVrJUU-PvKiQnipJKVtw
2. 向量体系(Embedding)在严选的落地实践
   https://mp.weixin.qq.com/s/NJDfrGJgIE2KK_t-yJ-C9Q
3.  [Embedding技术在房产推荐中的应用](https://mp.weixin.qq.com/s/flmPJtzeXLXDQXusI3Umxw) 
4.  [Embedding 技术在民宿推荐中的应用-201907](./content/精读-Embedding技术在民宿推荐中的应用-201907.md)  使用item2vec对app内房源进行embding，然后进行推荐，细节比较多，包括训练细节，数据构造细节等等，推荐看一看，我自己有精读，大家可以对照着看一看  精读完成     

### 3.5 用户画像

1. 网易严选画像建设实践
   https://mp.weixin.qq.com/s/pmovTV3TIoB6oA60pL_zeg

2. 网易大数据用户画像实践
   https://mp.weixin.qq.com/s/jyiDWiK0zczEaZKY5Hy5xg

3. 阅文用户画像
   https://mp.weixin.qq.com/s/ddRjNDBVuY03nQSGLncjtg

### 3.6 用户序列建模

1. 网易云音乐推荐中的用户行为序列深度建模
   https://mp.weixin.qq.com/s/Whf0rmuVapzZAB33TUj1Ig

## 4. NLP

### 4.1 内容理解

1. 腾讯信息流内容理解技术实践
   https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247495622&idx=1&sn=3d229e34dfe061b61bb47b4677def6a0&chksm=fbd75daacca0d4bc83d02b78b7d7c8485521eba07a03553db52ba039f3d948835f3d750a301e&scene=21#wechat_redirect

2. 微信"看一看"内容理解与推荐
   https://mp.weixin.qq.com/s/vr9bKEXI5o6L3FYig4HgNA
3. 腾讯信息流热点挖掘技术实践
   https://mp.weixin.qq.com/s/keSYVCS0k3rvngGvcsJKbA
4. 医疗健康领域的短文本理解-丁香园
   https://mp.weixin.qq.com/s/CNBP5xSvr4Y3Xm1-NMS79g
5. 热点挖掘技术在微信看一看中的应用
   https://mp.weixin.qq.com/s/oMNy-g2DxUnsGErefQBkyg
6.  [NLP 技术在微博 feed 流中的应用](https://www.infoq.cn/article/O5ytPDlYkfX3H26k6zru) 

 ### 4.2文本分类



### 4.3 文本匹配

1. [语义匹配在贝壳找房智能客服中的应用-202005](https://mp.weixin.qq.com/s?src=11&timestamp=1591783120&ver=2392&signature=RZJ5qcZ5PEc0eHDi9eznGXdaoQM2s2WEgsQgMlft5aPuOUiveyUcsoMCIm-sefmm8sRV2OpzrpsoaR6xAv8He0Q84azUJ5wv5gcvB1KQcx7OyN7A1b0QIt2xIpvhSSRH&new=1) 

2.  [丁香园在语义匹配任务上的探索与实践](https://mp.weixin.qq.com/s/Zn7oXWQPOt6KM1MsOtYfxA) 

3.  [深度文本表征与深度文本聚类在小样本场景中的探索与实践](https://mp.weixin.qq.com/s/dWAf2kczbjhQGhn8MZdjAQ) 

4. 爱奇艺深度语义表示学习的探索与实践
   https://mp.weixin.qq.com/s/f524bPx0pq7qxXGjpa7WCQ

   

### 4.4 命名体识别



### 4.5 预训练模型

1. 网易严选nlp-预训练语言模型的应用
https://mp.weixin.qq.com/s/hUbMbmEtLB7L0_H4DGc8Ew
2. 通用的图像-文本语言表征学习：多模态预训练模型 UNITER
    https://mp.weixin.qq.com/s/GxQ27vY5naaAXtp_ZTV0ZA
3.  [美团BERT的探索和实践-201911](https://tech.meituan.com/2019/11/14/nlp-bert-practice.html)  Bert在美团场景中的改进和优化，很厉害，细节很多          

### 4.6 文本生成

1. Seq2seq框架下的文本生成-丁香园
https://mp.weixin.qq.com/s/NAPIUtTD7ZEAIEgoJeQM4A

### 4.7 文本纠错

1. 基于BERT的ASR纠错
   https://mp.weixin.qq.com/s/JyXN9eukS-5XKvcJORTobg

### 4.8 数据增强

1. 文本增强技术的研究进展及应用实践
   https://mp.weixin.qq.com/s/CHSDi2LpDOLMjWOLXlvSAg

### 4.9 情感分析

1.  [情感分析算法在阿里小蜜的应用实践](https://mp.weixin.qq.com/s/k-gS6k3-hy-ZI_r901IGvg) 

### 4.10 智能客服

1. [AI技术如何打造58同城智能客服商家版“微聊管家”](https://mp.weixin.qq.com/s/_D9HX03ZmnXXX72MTrCx7g) 
2.  [58同城智能客服技术解密](https://mp.weixin.qq.com/s/5ewD2xD8J08W89-Rwixw4Q) 
3. 深度学习在网易严选智能客服中的应用
   https://mp.weixin.qq.com/s/SPtNy_1_6fiFXKukMmVPlA
4. 严选智能客服业务知识库自动挖掘方案
   https://mp.weixin.qq.com/s/AyaDkbKYhdRbc-uvFHNv4w

### 4.11 其他


## 5. 多模态：

1. [爱奇艺短视频分类技术解析](https://www.infoq.cn/article/f49e-Gb1xQxh8DttFDgb)  

全文重点在三个，一个是爱奇艺视频分类体系，一个是层次表示模块，一个是特征模块，介绍的比较详细，需要精读一下 
 [爱奇艺短视频质量评估模型](https://toutiao.io/posts/pbf8qf/preview)  

介绍封面文本视频内容质量评分，大概读一下就可以               

3. [阿里文娱多模态视频分类算法中的特征改进](https://www.6aiq.com/article/1585549128737)  

主要介绍类目体系构建介绍，模型特征调优，模型调优，值得精读一下 

4. [爱奇艺短视频软色情识别技术解析](https://www.infoq.cn/article/D7Ks_lLADmKFIm7ipMlP)  

很细致，看一下                       

5. [优酷在多模态内容理解上的研究及应用](https://www.infoq.cn/article/xgP_eyfidAA2l5ShcCPp)  

主要是一些多模态概念的讲解，干货不多，大概看一下就可以      

6. [多模态商品分类解决方案-深度学习在真实NAVER购物网站的应用](https://juejin.im/post/5d2dc7a9f265da1b96133852) 

 细节点比较多，偏实战，精读一下       

7. [FashionBERT 电商领域多模态研究：如何做图文拟合？](https://developer.aliyun.com/article/763357) 
8. 阿里文娱多模态视频分类算法中的特征改进
   https://mp.weixin.qq.com/s/6kTb6r3Vj3mgQn90UsZ1nw
9. 美团本地生活场景的短视频分析
   https://mp.weixin.qq.com/s/UyMDskA0eGN-NmiifwQF6Q
10. 爱奇艺视频精彩度分析
    https://mp.weixin.qq.com/s/hDqyItDxBfJ652BI0OT4dA

11. [5G 时代下：多模态理解做不到位注定要掉队](https://www.infoq.cn/article/EoEdfBO3-RNW1btNsQjJ) 

12. [短视频数据补充1](https://www.ctolib.com/yuanxiaosc-Multimodal-short-video-dataset-and-baseline-classification-model.html)[补充2](https://yuanxiaosc.github.io/2019/07/11/%E7%9F%AD%E8%A7%86%E9%A2%91%E5%88%86%E7%B1%BB%E6%8A%80%E6%9C%AF/) 

13. [UC信息流视频标签识别技术](https://www.secrss.com/articles/14055) 

14.  [技术动态-多模态学习调研 (附完整PPT）](https://mp.weixin.qq.com/s/g3rwPsusYi7gQopOHvdNrA) 

15. [多模态情感分析简述](https://mp.weixin.qq.com/s/xzeNAuuDt_VLHDgvIkc-Mg) 