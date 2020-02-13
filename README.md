# Awesome-VAEs
Awesome work on the VAE, disentanglement, representation learning, and generative models. 

I gathered these resources (currently @ 437 papers) as literature for my PhD, and thought it may come in useful for others. This list includes works relevant to various topics relating to VAEs. Sometimes this spills over to topics e.g. adversarial training and GANs, general disentanglement, variational inference, flow-based models and auto-regressive models. Always keen to expand the list.  I have also included an excel file which includes notes on each paper, as well as a breakdown of the topics covered in each paper.

They are ordered by year (new to old). I provide a link to the paper as well as to the github repo where available.


## 2020

Learning discrete and continuous factors of data via alternating disentanglement.	Jeong, Song	http://proceedings.mlr.press/v97/jeong19d/jeong19d.pdf https://github.com/snu-mllab/DisentanglementICML19


Electrocardiogram generation and feature extraction using a variational autoencoder. Kuznetsov, Moskalenko, Zolotykh	https://arxiv.org/pdf/2002.00254.pdf

CosmoVAE: variational autoencoder for CMB image inpainting.	Yi, Guo, Fan, Hamann, Wang	https://arxiv.org/pdf/2001.11651.pdf

Unsupervised representation disentanglement using cross domain features and adversarial learning in variational autoencoder based voice conversion.	Huang, Luo, Hwang, Lo, Peng, Tsao, Wang	https://arxiv.org/pdf/2001.07849.pdf

On implicit regularization in beta VAEs.	Kumar, Poole	https://arxiv.org/pdf/2002.00041.pdf

Weakly-supervised disentanglement without compromises.	Locatello, Poole, Ratsch, Scholkopf, Bachem, Tschannen	https://arxiv.org/pdf/2002.02886.pdf


An integrated framework based on latent variational autoencoder for providing early warning of at-risk students. 	Du, Yang, Hung	https://ieeexplore.ieee.org/abstract/document/8952699

Variational autoencoder and friends.	Zheng	https://www.cs.cmu.edu/~xunzheng/files/vae_single.pdf

High-fidelity synthesis with disentangled representation.	Lee, Kim, Hong, Lee	https://arxiv.org/pdf/2001.04296.pdf

Neurosymbolic knowledge representation for explainable and trustworthy AI.	Malo	https://www.preprints.org/manuscript/202001.0163/v1

Adversarial disentanglement with grouped observations.	Nemeth	https://arxiv.org/pdf/2001.04761.pdf

AE-OT-GAN: Training GANs from data specific latent distribution.	An, Guo, Zhang, Qi, Lei, Yau, Gu	https://arxiv.org/pdf/2001.03698.pdf

AE-OT: a new generative model based on extended semi-discrete optimal transport.	An, Guo, Lei, Luo, Yau, Gu	https://openreview.net/pdf?id=HkldyTNYwH

Disentanglement by nonlinear ICA with general incompressible-flow networks (GIN).	Sorrenson, Rother, Kothe	https://arxiv.org/pdf/2001.04872.pdf

Phase transitions for the information bottleneck in representation learning. 	Wu, Fischer	https://arxiv.org/pdf/2001.01878.pdf

Bayesian deep learning: a model-based interpretable approach. 	Matsubara	https://www.jstage.jst.go.jp/article/nolta/11/1/11_16/_article

SPACE: unsupervised object-oriented scene representation via spatial attention and decomposition. 	Lin, Wu, Peri, Sun, Singh, Deng, Jiang, Ahn	https://openreview.net/forum?id=rkl03ySYDH

A variational stacked autoencoder with harmony search optimizer for valve train fault diagnosis of diesel engine.	Chen, Mao, Zhao, Jiang, Zhang	https://www.mdpi.com/1424-8220/20/1/223

Evaluating loss compression rates of deep generative models.	anon 	https://openreview.net/forum?id=ryga2CNKDH

Progressive learning and disentanglement of hierarchical representations.	 anon	https://openreview.net/forum?id=SJxpsxrYPS

## 2019

Group-based learning of disentangled representations with generalizability for novel contents.	Hosoya	https://www.ijcai.org/Proceedings/2019/0348.pdf

Task-Conditioned variational autoencoders for learning movement primitives.	Noseworthy, Paul, Roy, Park, Roy	https://groups.csail.mit.edu/rrg/papers/noseworthy_corl_19.pdf

Multimodal generative models for compositional representation learning.	Wu, Goodman	https://arxiv.org/pdf/1912.05075.pdf

dpVAEs: fixing sample generation for regularized VAEs.	Bhalodia, Lee, Elhabian	https://arxiv.org/pdf/1911.10506.pdf

From variational to deterministic autoencoders. 	Ghosh, Sajjadi, Vergai, Black, Scholkopf	https://arxiv.org/pdf/1903.12436.pdf

Learning representations by maximizing mutual information in variational autoencoder.	Rezaabad, Vishwanath	https://arxiv.org/pdf/1912.13361.pdf	

Disentangled representation learning with Wasserstein total correlation.	Xiao, Wang	https://arxiv.org/pdf/1912.12818.pdf	

Wasserstein dependency measure for representation learning. 	Ozair, Lynch, Bengio, van den Oord, Levine, Sermanent	https://arxiv.org/pdf/1903.11780.pdf	

GP-VAE: deep probabilistic time series imputation.	Fortuin, Baranchuk, Ratsch, Mandt	https://arxiv.org/pdf/1907.04155.pdf	https://github.com/ratschlab/GP-VAE

Likelihood contribution based multi-scale architecture for generative flows.	Das, Abbeel, Spanos	 https://arxiv.org/pdf/1908.01686.pdf	

Gated Variational Autoencoders: Incorporating weak supervision to encourage disentanglement.	Vowels, Camgoz, Bowden	 https://arxiv.org/pdf/1911.06443.pdf	

An introduction to variational autoencoders.	Kingma, Welling	https://arxiv.org/pdf/1906.02691.pdf

Adaptive density estimation for generative models	Lucas, Shmelkov, Schmid, Alahari, Verbeek	https://papers.nips.cc/paper/9370-adaptive-density-estimation-for-generative-models.pdf

Data efficient mutual information neural estimator	Lin, Sur, Nastase, Divakaran, Hasson, Amer	https://arxiv.org/pdf/1905.03319.pdf

RecVAE: a new variational autoencoder for Top-N recommendations with implicit feedback.	Shenbin, Alekseev, Tutubalina, Malykh, Nikolenko	https://arxiv.org/pdf/1912.11160.pdf

Vibration signal generation using conditional variational autoencoder for class imbalance problem.	Ko, Kim, Kong, Lee, Youn	http://icmr2019.ksme.or.kr/wp/pdf/190090.pdf

The usual suspects? Reassessing blame for VAE posterior collapse. 	Dai, Wang, Wipf	https://arxiv.org/pdf/1912.10702.pdf

What does the free energy principle tell us about the brain?	Gershman	https://arxiv.org/pdf/1901.07945.pdf

Sub-band vector quantized variational autoencoder for spectral envelope quantization.	Srikotr, Mano	https://ieeexplore.ieee.org/abstract/document/8929436	

A variational-sequential graph autoencoder for neural performance prediction.	Friede, Lukasik, Stuckenschmidt, Keuper	https://arxiv.org/pdf/1912.05317.pdf	

Explicit disentanglement of appearance and perspective in generative models.	Skafte, Hauberg	https://papers.nips.cc/paper/8387-explicit-disentanglement-of-appearance-and-perspective-in-generative-models.pdf	

Disentangled behavioural representations.	Dezfouli, Ashtiani, Ghattas, Nock, Dayan, Ong	https://papers.nips.cc/paper/8497-disentangled-behavioural-representations.pdf	

Learning disentangled representations for robust person re-identification.	Eom, Ham	https://papers.nips.cc/paper/8771-learning-disentangled-representation-for-robust-person-re-identification.pdf	

Towards latent space optimality for auto-encoder based generative models.	Mondal, Chowdhury, Jayendran, Singla, Asnani, AP	https://arxiv.org/pdf/1912.04564.pdf	

Don't blame the ELBO! A linear VAE perspective on posterior collapse.	Lucas, Tucker, Grosse, Norouzi	https://128.84.21.199/pdf/1911.02469.pdf

Bridging the ELBO and MMD.	Ucar	https://arxiv.org/pdf/1910.13181.pdf

Learning disentangled representations for counterfactual regression. 	Hassanpour, Greiner	https://pdfs.semanticscholar.org/1df4/204e14da51b05a14781e2a4dc3e0d7da562d.pdf

Learning disentangled representations for recommendation.	Ma, Zhou, Cui, Yang, Zhu	https://arxiv.org/pdf/1910.14238.pdf

A vector quantized variational autoencoder (VQ-VAE) autoregressive neural F0 model for statistical parametric speech synthesis.	Wang, Takaki, Yamagishi, King, Tokuda	https://ieeexplore.ieee.org/abstract/document/8884734

Diversity-aware event prediction based on a conditional variational autoencoder with reconstruction.	Kiyomaru, Omura, Murawaki, Kawahara, Kurohashi	https://www.aclweb.org/anthology/D19-6014.pdf

Learning multimodal representations with factorized deep generative models.	Tsai, Liang, Zadeh, Morency, Salakhutdinov	https://pdfs.semanticscholar.org/7416/6384ad391513e8e8bf48cbeaff2516b8c332.pdf

High-dimensional nonlinear profile monitoring based on deep probabilistic autoencoders.	Sergin, Yan	https://arxiv.org/pdf/1911.00482.pdf

Leveraging directed causal discovery to detect latent common causes.	Lee, Hart, Richens, Johri	https://arxiv.org/pdf/1910.10174.pdf

Robust discrimination and generation of faces using compact, disentangled embeddings.	Browatzki, Wallraven	http://openaccess.thecvf.com/content_ICCVW_2019/papers/RSL-CV/Browatzki_Robust_Discrimination_and_Generation_of_Faces_using_Compact_Disentangled_Embeddings_ICCVW_2019_paper.pdf

Coulomb Autoencoders.	Sansone, Ali, Sun	https://arxiv.org/pdf/1802.03505.pdf

Contrastive learning of structured world models. 	Kipf, Pol, Welling	https://arxiv.org/pdf/1911.12247.pdf

No representation without transformation.	Giannone, Masci, Osendorfer	https://pgr-workshop.github.io/img/PGR007.pdf

Neural density estimation. 	Papamakarios	https://arxiv.org/pdf/1910.13233.pdf

Variational autoencoder-based approach for rail defect identification. 	Wei, Ni	http://www.dpi-proceedings.com/index.php/shm2019/article/view/32432

Variational learning with disentanglement-pytorch.	Abdi, Abolmaesumi, Fels	https://openreview.net/pdf?id=rJgUsFYnir

PVAE: learning disentangled representations with intrinsic dimension via approximated L0 regularization.	Shi, Glocker, Castro	https://openreview.net/pdf?id=HJg8stY2oB

Mixed-curvature variational autoencoders. 	Skopek, Ganea, Becigneul	https://arxiv.org/pdf/1911.08411.pdf

Continuous hierarchical representations with poincare variational autoencoders. 	Mathieu, Le Lan, Maddison, Tomioka	https://arxiv.org/pdf/1901.06033.pdf

VIREL: A variational inference framework for reinforcement learning.	Fellows, Mahajan, Rudner, Whiteson	https://arxiv.org/pdf/1811.01132.pdf

Disentangling video with independent prediction.	Whitney, Fergus	https://arxiv.org/pdf/1901.05590.pdf

Disentangling state space representations	Miladinovic, Gondal, Scholkopf, Buhmann, Bauer	https://arxiv.org/pdf/1906.03255.pdf

Likelihood conribution based multi-scale architecture for generative flows.	Das, Abbeel, Spanos	https://arxiv.org/pdf/1908.01686.pdf

AlignFlow: cycle consistent learning from multiple domains via normalizing flows	Grover, Chute, Shu, Cao, Ermon	https://arxiv.org/pdf/1905.12892.pdf


IB-GAN: disentangled representation learning with information bottleneck GAN. 	Jeon, Lee, Kim	https://openreview.net/forum?id=ryljV2A5KX

Learning hierarchical priors in VAEs.	 Klushyn, Chen, Kurle, Cseke, van der Smagt	https://papers.nips.cc/paper/8553-learning-hierarchical-priors-in-vaes.pdf

ODE2VAE: Deep generative second order ODEs with Bayesian neural networks.	Yildiz, Heinonen, Lahdesmaki	https://papers.nips.cc/paper/9497-ode2vae-deep-generative-second-order-odes-with-bayesian-neural-networks.pdf

Explicitly disentangling image content from translation and rotation with spatial-VAE.	Bepler, Zhong, Kelley, Brignole, Berger	https://papers.nips.cc/paper/9677-explicitly-disentangling-image-content-from-translation-and-rotation-with-spatial-vae.pdf

A primal-dual link between GANs and autoencoders.	Husain, Nock, Williamson	https://papers.nips.cc/paper/8333-a-primal-dual-link-between-gans-and-autoencoders.pdf

Exact rate-distortion in autoencoders via echo noise.	Brekelmans, Moyer, Galstyan, ver Steeg	https://papers.nips.cc/paper/8644-exact-rate-distortion-in-autoencoders-via-echo-noise.pdf

Direct optimization through arg max for discrete variational auto-encoder.	Lorberbom, Jaakkola, Gane, Hazan	https://papers.nips.cc/paper/8851-direct-optimization-through-arg-max-for-discrete-variational-auto-encoder.pdf

Semi-implicit graph variational auto-encoders.	Hasanzadeh, Hajiramezanali, Narayanan, Duffield, Zhou, Qian	https://papers.nips.cc/paper/9255-semi-implicit-graph-variational-auto-encoders.pdf

The continuous Bernoulli: fixing a pervasive error in variational autoencoders.	Loaiza-Ganem, Cunningham	https://papers.nips.cc/paper/9484-the-continuous-bernoulli-fixing-a-pervasive-error-in-variational-autoencoders.pdf

Provable gradient variance guarantees for black-box variational inference. 	Domke	https://papers.nips.cc/paper/8325-provable-gradient-variance-guarantees-for-black-box-variational-inference.pdf

Conditional structure generation through graph variational generative adversarial nets.	Yang, Zhuang, Shi, Luu, Li	https://papers.nips.cc/paper/8415-conditional-structure-generation-through-graph-variational-generative-adversarial-nets.pdf

Scalable spike source localization in extracellular recordings using amortized variational inference.	Hurwitz, Xu, Srivastava, Buccino, Hennig	https://papers.nips.cc/paper/8720-scalable-spike-source-localization-in-extracellular-recordings-using-amortized-variational-inference.pdf

A latent variational framework for stochastic optimization.	Casgrain	https://papers.nips.cc/paper/8802-a-latent-variational-framework-for-stochastic-optimization.pdf

MAVEN: multi-agent variational exploration.	Mahajan, Rashid, Samvelyan, Whiteson	https://papers.nips.cc/paper/8978-maven-multi-agent-variational-exploration.pdf

Variational graph recurrent neural networks.	Hajiramezanali, Hasanzadeh, Narayanan, Duffield, Zhou, Qian	https://papers.nips.cc/paper/9254-variational-graph-recurrent-neural-networks.pdf

The thermodynamic variational objective.	Masrani, Le, Wood	https://papers.nips.cc/paper/9328-the-thermodynamic-variational-objective.pdf

Variational temporal abstraction. 	Kim, Ahn, Bengio	https://papers.nips.cc/paper/9332-variational-temporal-abstraction.pdf

Exploiting video sequences for unsupervised disentangling in generative adversarial networks.	Tuesca, Uzal	https://arxiv.org/pdf/1910.11104.pdf

Couple-VAE: mitigating the encoder-decoder incompatibility in variational text modeling with coupled deterministic networks.		https://openreview.net/pdf?id=SJlo_TVKwS

Variational mixture-of-experts autoencoders for multi-modal deep generative models. Shi, Siddharth, Paige, Torr	https://papers.nips.cc/paper/9702-variational-mixture-of-experts-autoencoders-for-multi-modal-deep-generative-models.pdf

Invertible convolutional flow.	Karami, Schuurmans, Sohl-Dickstein, Dinh, Duckworth	https://papers.nips.cc/paper/8801-invertible-convolutional-flow.pdf

Implicit posterior variational inference for deep Gaussian processes.	Yu, Chen, Dai, Low, Jaillet	https://papers.nips.cc/paper/9593-implicit-posterior-variational-inference-for-deep-gaussian-processes.pdf

MaCow: Masked convolutional generative flow.	Ma, Kong, Zhang, Hovy	https://papers.nips.cc/paper/8824-macow-masked-convolutional-generative-flow.pdf

Residual flows for invertible generative modeling.	Chen, Behrmann, Duvenaud, Jacobsen	https://papers.nips.cc/paper/9183-residual-flows-for-invertible-generative-modeling.pdf

Discrete flows: invertible generative models of discrete data.	Tran, Vafa, Agrawal, Dinh, Poole	https://papers.nips.cc/paper/9612-discrete-flows-invertible-generative-models-of-discrete-data.pdf

Re-examination of the role of latent variables in sequence modeling. 	Lai, Dai, Yang, Yoo	https://papers.nips.cc/paper/8996-re-examination-of-the-role-of-latent-variables-in-sequence-modeling.pdf

Learning-in-the-loop optimization: end-to-end control and co-design of soft robots through learned deep latent representations.	Spielbergs, Zhao, Hu, Du, Matusik, Rus	https://papers.nips.cc/paper/9038-learning-in-the-loop-optimization-end-to-end-control-and-co-design-of-soft-robots-through-learned-deep-latent-representations.pdf

Triad constraints for learning causal structure of latent variables.	Cai, Xie, Glymour, Hao, Zhang	https://papers.nips.cc/paper/9448-triad-constraints-for-learning-causal-structure-of-latent-variables.pdf

Disentangling influence: using disentangled representations to audit model predictions.	Marx, Phillips, Friedler, Scheidegger, Venkatasubramanian	https://papers.nips.cc/paper/8699-disentangling-influence-using-disentangled-representations-to-audit-model-predictions.pdf

Symmetry-based disentangled representation learning requires interaction with environments.	Caselles-Dupre, Ortiz, Filliat	https://papers.nips.cc/paper/8709-symmetry-based-disentangled-representation-learning-requires-interaction-with-environments.pdf

Weakly supervised disentanglement with guarantees.	Shu, Chen, Kumar, Ermon, Poole	https://arxiv.org/pdf/1910.09772.pdf

Demystifying inter-class disentanglement.	Gabbay, Hoshen	https://arxiv.org/pdf/1906.11796.pdf

Spectral regularization for combating mode collapse in GANs.	Liu, Tang, Xie, Qiu	https://arxiv.org/pdf/1908.10999.pdf

Geometric disentanglement for generative latent shape models.	Aumentado-Armstrong, Tsogkas, Jepson, Dickinson	https://arxiv.org/pdf/1908.06386.pdf

Cross-dataset person re-identification via unsupervised pose disentanglement and adaptation.	Li, Lin, Lin, Wang	https://arxiv.org/pdf/1909.09675.pdf

Identity from here, pose from there: self-supervised disentanglement and generation of objects using unlabeled videos.	Xiao, Liu, Lee	https://web.cs.ucdavis.edu/~yjlee/projects/iccv2019_disentangle.pdf

Content and style disentanglement for artistic style transfer.	Kotovenko, Sanakoyeu, Lang, Ommer	https://compvis.github.io/content-style-disentangled-ST/paper.pdf

Unsupervised robust disentangling of latent characteristics for image synthesis.	Esser, Haux, Ommer	https://arxiv.org/pdf/1910.10223.pdf

LADN: local adversarial disentangling network for facial makeup and de-makeup.	Gu, Wang, Chiu, Tai, Tang	https://arxiv.org/pdf/1904.11272.pdf

Video compression with rate-distortion autoencoders.	Habibian, van Rozendaal, Tomczak, Cohen	https://arxiv.org/pdf/1908.05717.pdf

Variable rate deep image compression with a conditional autoencoder.	Choi, El-Khamy, Lee	https://arxiv.org/pdf/1909.04802.pdf

Memorizing normality to detect anomaly: memory-augmented deep autoencoder for unsupervised anomaly detection.	Gong, Liu, Le, Saha	https://arxiv.org/pdf/1904.02639.pdf

AVT: unsupervise d learning of transformation equivariant representations by autoencoding variational transformations.	Qi, Zhang, Chen, Tian	https://arxiv.org/pdf/1903.10863.pdf

Deep clustering by Gaussian mixture variational autoencoders with graph embedding.	Yang, Cheung, Li, Fang	http://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_Deep_Clustering_by_Gaussian_Mixture_Variational_Autoencoders_With_Graph_Embedding_ICCV_2019_paper.pdf

Variational adversarial active learning.	Sinha, Ebrahimi, Darrell	https://arxiv.org/pdf/1904.00370.pdf

Variational few-shot learning.	Zhang, Zhao, Ni, Xu, Yang	http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Variational_Few-Shot_Learning_ICCV_2019_paper.pdf

Multi-angle point cloud-VAE: unsupervised feature learning for 3D point clouds from multiple angles by joint self-reconstruction and half-to-half prediction.	Han, Wang, Liu, Zwicker	https://arxiv.org/pdf/1907.12704.pdf

LayoutVAE: stochastic scene layout generation from a label set.	 Jyothi, Durand, He, Sigal, Mori	https://arxiv.org/pdf/1907.10719.pdf

VV-NET: Voxel VAE Net with group convolutions for point cloud segmentation.	Meng, Gao, Lai, Manocha	https://arxiv.org/pdf/1811.04337.pdf

Bayes-Factor-VAE: hierarchical bayesian deep auto-encoder models for factor disentanglement.	Kim, Wang, Sahu, Pavlovic	https://arxiv.org/pdf/1909.02820.pdf


Robust ordinal VAE: Employing noisy pairwise comparisons for disentanglement.	Chen, Batmanghelich	https://arxiv.org/pdf/1910.05898.pdf

Evaluating disentangled representations.	Sepliarskaia, A. and Kiseleva, J. and de Rijke, M.	https://arxiv.org/pdf/1910.05587.pdf

A stable variational autoencoder for text modelling.	Li, R. and Li, X. and Lin, C. and Collinson, M. and Mao, R.	https://abdn.pure.elsevier.com/en/publications/a-stable-variational-autoencoder-for-text-modelling

Hamiltonian generative networks.	Toth, Rezende, Jaegle, Racaniere, Botev, Higgins	https://128.84.21.199/pdf/1909.13789.pdf

LAVAE: Disentangling location and appearance. Dittadi, Winther	https://arxiv.org/pdf/1909.11813.pdf

Interpretable models in probabilistic machine learning.	Kim	https://ora.ox.ac.uk/objects/uuid:b238ed7d-7155-4860-960e-6227c7d688fb/download_file?file_format=pdf&safe_filename=PhD_Thesis_of_University_of_Oxford.pdf&type_of_work=Thesis


Disentangling speech  and non-speech components for building robust acoustic models from found data.	Gurunath, Rallabandi, Black	https://arxiv.org/pdf/1909.11727.pdf

Joint separation, dereverberation and classification of multiple sources using multichannel variational autoencoder with auxiliary classifier.	Inoue, Kameoka, Li, Makino	http://pub.dega-akustik.de/ICA2019/data/articles/000906.pdf

SuperVAE: Superpixelwise variational autoencoder for salient object detection.	Li, Sun, Guo	https://www.aaai.org/ojs/index.php/AAAI/article/view/4876

Implicit discriminator in variational autoencoder. 	Munjal, Paul, Krishnan	https://arxiv.org/pdf/1909.13062.pdf

TransGaGa: Geometry-aware unsupervised image-to-image translation.	Wu, Cao, Li, Qian, Loy	http://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_TransGaGa_Geometry-Aware_Unsupervised_Image-To-Image_Translation_CVPR_2019_paper.pdf

Variational attention using articulatory priors for generating code mixed speech using monolingual corpora.	Rallabandi, Black.	https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1103.pdf

One-class collaborative filtering with the queryable variational autoencoder.	Wu, Bouadjenek, Sanner.	https://people.eng.unimelb.edu.au/mbouadjenek/papers/SIGIR_Short_2019.pdf

Predictive auxiliary variational autoencoder for representation learning of global speech characteristics.	Springenberg, Lakomkin, Weber, Wermter.	https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2845.pdf

Data augmentation using variational autoencoder for embedding based speaker verification.	Wu, Wang, Qian, Yu	https://zhanghaowu.me/assets/VAE_Data_Augmentation_proceeding.pdf

One-shot voice conversion with disentangled representations by leveraging phonetic posteriograms.	Mohammadi, Kim.	https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1798.pdf

EEG-based adaptive driver-vehicle interface using variational autoencoder and PI-TSVM.	Bi, Zhang, Lian	https://www.researchgate.net/profile/Luzheng_Bi2/publication/335619300_EEG-Based_Adaptive_Driver-Vehicle_Interface_Using_Variational_Autoencoder_and_PI-TSVM/links/5d70bb234585151ee49e5a30/EEG-Based-Adaptive-Driver-Vehicle-Interface-Using-Variational-Autoencoder-and-PI-TSVM.pdf

Neural gaussian copula for variational autoencoder	Wang, Wang	https://arxiv.org/pdf/1909.03569.pdf

Enhancing VAEs for collaborative filtering: Flexible priors and gating mechanisms.	Kim, Suh	http://delivery.acm.org/10.1145/3350000/3347015/p403-kim.pdf?ip=86.162.136.199&id=3347015&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1568726810_89cfa7cbc7c1b0663405d4446f9fce85

Riemannian normalizing flow on variational wasserstein autoencoder for text modeling.	Wang, Wang	https://arxiv.org/pdf/1904.02399.pdf

Disentanglement with hyperspherical latent spaces using diffusion variational autoencoders.	Rey	https://openreview.net/pdf?id=SylFDSU6Sr

Learning deep representations by mutual information estimation and maximization.	Hjelm, Fedorov, Lavoie-Marchildon, Grewal, Bachman, Trischler, Bengio	https://arxiv.org/pdf/1808.06670.pdf	https://github.com/rdevon/DIM

Novel tracking approach based on fully-unsupervised disentanglement of the geometrical factors of variation.	Vladymyrov, Ariga	https://arxiv.org/pdf/1909.04427.pdf

Real time trajectory prediction using conditional generative models.	Gomez-Gonzalez, Prokudin, Scholkopf, Peters	https://arxiv.org/pdf/1909.03895.pdf

Disentanglement challenge: from regularization to reconstruction.	Qiao, Li, Cai	https://openreview.net/pdf?id=ByecPrUaHH

Improved disentanglement through aggregated convolutional feature maps.	Seitzer	https://openreview.net/pdf?id=ryxOvH86SH

Linked variational autoencoders for inferring substitutable and supplementary items.	Rakesh, Wang, Shu	http://www.public.asu.edu/~skai2/files/wsdm_2019_lvae.pdf

On the fairness of disentangled representations.	Locatello, Abbati, Rainforth, Bauer, Scholkopf, Bachem	https://arxiv.org/pdf/1905.13662.pdf

Learning robust representations by projecting superficial statistics out.	Wang, He, Lipton, Xing	https://openreview.net/pdf?id=rJEjjoR9K7

Understanding posterior collapse in generative latent variable models.	Lucas, Tucker, Grosse, Norouzi	https://openreview.net/pdf?id=r1xaVLUYuE

On the transfer of inductive bias from simulation to the real world: a new disentanglement dataset.	Gondal, Wuthrich, Miladinovic, Locatello, Breidt, Volchkv, Akpo, Bachem, Scholkopf, Bauer	https://arxiv.org/pdf/1906.03292.pdf	https://github.com/rr-learning/disentanglement_dataset

DIVA: domain invariant variational autoencoder.	Ilse, Tomczak, Louizos, Welling	https://arxiv.org/pdf/1905.10427.pdf	https://github.com/AMLab-Amsterdam/DIVA

Comment: Variational Autoencoders as empirical Bayes.	Wang, Miller, Blei	http://www.stat.columbia.edu/~yixinwang/papers/WangMillerBlei2019.pdf	

Fast MVAE: joint separation and classification of mixed sources based on multichannel variational autoencoder with auxiliary classifier.	Li, Kameoka, Makino	https://ieeexplore.ieee.org/abstract/document/8682623	

Reweighted expectation maximization.	Dieng, Paisley	https://arxiv.org/pdf/1906.05850.pdf	https://github.com/adjidieng/REM

Semisupervised text classification by variational autoencoder.	Xu, Tan	https://ieeexplore.ieee.org/abstract/document/8672806	

Learning deep latent-variable MRFs with amortized Bethe free-energy minimization.	Wiseman	https://openreview.net/pdf?id=ByeMHULt_N

Contrastive variational autoencoder enhances salient features.	Abid, Zou	https://arxiv.org/pdf/1902.04601.pdf	https://github.com/abidlabs/contrastive_vae

Learning latent superstructures in variational autoencoders for deep multidimensional clustering.	Li, Chen, Poon, Zhang	https://openreview.net/pdf?id=SJgNwi09Km


Tighter variational bounds are not necessarily better.	Rainforth, Kosiorek, Le, Maddison, Igl, Wood, The	https://arxiv.org/pdf/1802.04537.pdf	https://github.com/lxuechen/DReG-PyTorch

ISA-VAE: Independent subspace analysis with variational autoencoders. Anon. https://openreview.net/pdf?id=rJl_NhR9K7


Manifold mixup: better representations by interpolating hidden states. Verma, Lamb, Beckham, Najafi, Mitliagkas, Courville, Lopez-Paz, Bengio. 
https://arxiv.org/pdf/1806.05236.pdf 
https://github.com/vikasverma1077/manifold_mixup

Bit-swap: recursive bits-back coding for lossless compression with hierarchical latent variables. Kingma, Abbeel, Ho. http://proceedings.mlr.press/v97/kingma19a/kingma19a.pdf 
https://github.com/fhkingma/bitswap

Practical lossless compression with latent variables using bits back coding.	Townsend, Bird, Barber.	https://arxiv.org/pdf/1901.04866.pdf	https://github.com/bits-back/bits-back

BIVA:  a very deep hierarchy of latent variables for generative modeling.	Maaloe, Fraccaro, Lievin, Winther.	https://arxiv.org/pdf/1902.02102.pdf

Flow++: improving flow-based generative models with variational dequantization and architecture design.	Ho, Chen, Srinivas, Duan, Abbeel.	https://arxiv.org/pdf/1902.00275.pdf	https://github.com/aravindsrinivas/flowpp

Sylvester normalizing flows for variational inference.	van den Berg, Hasenclever, Tomczak, Welling.	https://arxiv.org/pdf/1803.05649.pdf	https://github.com/riannevdberg/sylvester-flows

Unbiased implicit variational inference.	Titsias, Ruiz.	https://arxiv.org/pdf/1808.02078.pdf

Robustly disentangled causal mechanisms: validating deep representations for interventional robustness. 	Suter, Miladinovic, Scholkopf, Bauer.	https://arxiv.org/pdf/1811.00007.pdf

Tutorial: Deriving the standard variational autoencoder (VAE) loss function.	Odaibo	https://arxiv.org/pdf/1907.08956.pdf

Learning disentangled representations with reference-based variational autoencoders.	Ruiz, Martinez, Binefa, Verbeek.	https://arxiv.org/pdf/1901.08534

Disentangling factors of variation using few labels.	Locatello, Tschannen, Bauer, Ratsch, Scholkopf, Bachem	https://arxiv.org/pdf/1905.01258.pdf	

Disentangling disentanglement in variational autoencoders	Mathieu, Rainforth, Siddharth, The,	https://arxiv.org/pdf/1812.02833.pdf	https://github.com/iffsid/disentangling-disentanglement

LIA: latently invertible autoencoder with adversarial learning	Zhu, Zhao, Zhang	https://arxiv.org/pdf/1906.08090.pdf	

Emerging disentanglement in auto-encoder based unsupervised image content transfer.	Press, Galanti, Benaim, Wolf	https://openreview.net/pdf?id=BylE1205Fm	https://github.com/oripress/ContentDisentanglement

MAE: Mutual posterior-divergence regularization for variational autoencoders	Ma, Zhou, Hovy	https://arxiv.org/pdf/1901.01498.pdf	https://github.com/XuezheMax/mae

Overcoming the disentanglement vs reconstruction trade-off via Jacobian supervision.	Lezama	https://openreview.net/pdf?id=Hkg4W2AcFm	https://github.com/jlezama/disentangling-jacobian https://github.com/jlezama/disentangling-jacobian/tree/master/unsupervised_disentangling

Challenging common assumptions in the unsupervised learning of disentangled representations.	Locatello, Bauer, Lucic, Ratsch, Gelly, Scholkopf, Bachem	https://arxiv.org/abs/1811.12359	https://github.com/google-research/disentanglement_lib/blob/master/README.md

Variational prototyping encoder: one shot learning with prototypical images.	Kim, Oh, Lee, Pan, Kweon	http://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Variational_Prototyping-Encoder_One-Shot_Learning_With_Prototypical_Images_CVPR_2019_paper.pdf	

Diagnosing and enchanving VAE models (conf and journal paper both available).	Dai, Wipf	https://arxiv.org/pdf/1903.05789.pdf	https://github.com/daib13/TwoStageVAE

Disentangling latent hands for image synthesis and pose estimation.	Yang, Yao	http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Disentangling_Latent_Hands_for_Image_Synthesis_and_Pose_Estimation_CVPR_2019_paper.pdf	

Rare event detection using disentangled representation learning.	Hamaguchi, Sakurada, Nakamura	http://openaccess.thecvf.com/content_CVPR_2019/papers/Hamaguchi_Rare_Event_Detection_Using_Disentangled_Representation_Learning_CVPR_2019_paper.pdf	

Disentangling latent space for VAE by label relevant/irrelvant dimensions.	Zheng, Sun	https://arxiv.org/pdf/1812.09502.pdf	https://github.com/ZhilZheng/Lr-LiVAE

Variational autoencoders pursue PCA directions (by accident).	Rolinek, Zietlow, Martius	https://arxiv.org/pdf/1812.06775.pdf	

Disentangled Representation learning for 3D face shape.	Jiang, Wu, Chen, Zhang	https://arxiv.org/pdf/1902.09887.pdf	https://github.com/zihangJiang/DR-Learning-for-3D-Face

Preventing posterior collapse with delta-VAEs.	Razavi, van den Oord, Poole, Vinyals	https://arxiv.org/pdf/1901.03416.pdf	https://github.com/mattjj/svae

Gait recognition via disentangled representation learning.	Zhang, Tran, Yin, Atoum, Liu, Wan, Wang	https://arxiv.org/pdf/1904.04925.pdf	

Hierarchical disentanglement of discriminative latent features for zero-shot learning.	Tong, Wang, Klinkigt, Kobayashi, Nonaka
http://openaccess.thecvf.com/content_CVPR_2019/papers/Tong_Hierarchical_Disentanglement_of_Discriminative_Latent_Features_for_Zero-Shot_Learning_CVPR_2019_paper.pdf	

Generalized zero- and few-shot learning via aligned variational autoencoders.	Schonfeld, Ebrahimi, Sinha, Darrell, Akata	https://arxiv.org/pdf/1812.01784.pdf	https://github.com/chichilicious/Generalized-Zero-Shot-Learning-via-Aligned-Variational-Autoencoders

Unsupervised part-based disentangling of object shape and appearance.	Lorenz, Bereska, Milbich, Ommer	https://arxiv.org/pdf/1903.06946.pdf	

A semi-supervised Deep generative model for human body analysis.	de Bem, Ghosh, Ajanthan, Miksik, Siddaharth, Torr	http://www.robots.ox.ac.uk/~tvg/publications/2018/W21P20.pdf

Multi-object representation learning with iterative variational inference.	Greff, Kaufman, Kabra, Watters, Burgess, Zoran, Matthey, Botvinick, Lerchner	https://arxiv.org/pdf/1903.00450.pdf	https://github.com/MichaelKevinKelly/IODINE

Generating diverse high-fidelity images with VQ-VAE-2.	Razavi, van den Oord, Vinyals	https://arxiv.org/pdf/1906.00446.pdf	https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb https://github.com/rosinality/vq-vae-2-pytorch


MONet: unsupervised scene decomposition and representation.	Burgess, Matthey, Watters, Kabra, Higgins, Botvinick, Lerchner	https://arxiv.org/pdf/1901.11390.pdf	

Structured disentangled representations and Hierarchical disentangled representations.	Esmaeili, Wu, Jain, Bozkurt, Siddarth, Paige, Brooks, Dy, van de Meent	https://arxiv.org/pdf/1804.02086.pdf	

Spatial Broadcast Decoder: A  simple architecture for learning disentangled representations in VAEs.	Watters, Matthey, Burgess, Lerchner	https://arxiv.org/pdf/1901.07017.pdf	https://github.com/lukaszbinden/spatial-broadcast-decoder

Resampled priors for variational autoencoders.	Bauer, Mnih	https://arxiv.org/pdf/1802.06847.pdf

Weakly supervised disentanglement by pairwise similiarities.	Chen, Batmanghelich	https://arxiv.org/pdf/1906.01044.pdf	

Deep variational information bottleneck.	Aelmi, Fischer, Dillon, Murphy	https://arxiv.org/pdf/1612.00410.pdf	https://github.com/alexalemi/vib_demo

Generalized variational inference.	Knoblauch, Jewson, Damoulas	https://arxiv.org/pdf/1904.02063.pdf	

Variational autoencoders and nonlinear ICA: a unifying framework.	Khemakhem, Kingma	https://arxiv.org/pdf/1907.04809.pdf	

Lagging inference networks and posterior collapse in variational autoencoders.	He, Spokoyny, Neubig, Berg-Kirkpatrick	https://arxiv.org/pdf/1901.05534.pdf	https://github.com/jxhe/vae-lagging-encoder

Avoiding latent variable collapse with generative skip models.	Dieng, Kim, Rush, Blei	https://arxiv.org/pdf/1807.04863.pdf	

Distribution Matching in Variational inference.	Rosca, Lakshminarayana, Mohamed	https://arxiv.org/pdf/1802.06847.pdf	
A variational auto-encoder model for stochastic point process.	Mehrasa, Jyothi, Durand, He, Sigal, Mori	https://arxiv.org/pdf/1904.03273.pdf	

Sliced-Wasserstein auto-encoders.	Kolouri, Pope, Martin, Rohde	https://openreview.net/pdf?id=H1xaJn05FQ	https://github.com/skolouri/swae

A deep generative model for graph layout.	Kwon, Ma	https://arxiv.org/pdf/1904.12225.pdf	

Differentiable perturb-and-parse semi-supervised parsing with a structured variational autoencoder.	Corro, Titov	https://openreview.net/pdf?id=BJlgNh0qKQ	https://github.com/FilippoC/diffdp

Variational autoencoders with jointly optimized latent dependency structure.	He, Gong, Marino, Mori, Lehrmann	https://openreview.net/pdf?id=SJgsCjCqt7	https://github.com/ys1998/vae-latent-structure

Unsupervised learning of spatiotemporally coherent metrics	Goroshin, Bruna, Tompson, Eigen, LeCun	https://arxiv.org/pdf/1412.6056.pdf

Temporal difference variational auto-encoder.	Gregor, Papamakarios, Besse, Buesing, Weber	https://arxiv.org/pdf/1806.03107.pdf	https://github.com/xqding/TD-VAE

Representation learning with contrastive predictive coding.	van den Oord, Li, Vinyals	https://arxiv.org/pdf/1807.03748.pdf	https://github.com/davidtellez/contrastive-predictive-coding

Representation disentanglement  for multi-task learning with application to fetal ultrasound	Meng, Pawlowski, Rueckert, Kainz	https://arxiv.org/pdf/1908.07885.pdf

M$2$VAE - derivation of a multi-modal variational autoencoder objective from the marginal joint log-likelihood.	Korthals	https://arxiv.org/pdf/1903.07303.pdf

Predicting visual memory schemas with variational autoencoders.	Kyle-Davidson, Bors, Evans	https://arxiv.org/pdf/1907.08514.pdf

T-CVAE: Transformer -based conditioned variational autoencoder for story completion.	Wang, Wan	https://www.ijcai.org/proceedings/2019/0727.pdf	https://github.com/sodawater/T-CVAE

PuVAE: A variational autoencoder to purify adversarial examples.	Hwang, Park, Jang, Yoon, Cho	https://arxiv.org/pdf/1903.00585.pdf

Coupled VAE: Improved accuracy and robustness of a variational autoencoder. 	Cao, Li, Nelson	https://arxiv.org/pdf/1906.00536.pdf

D-VAE: A variational autoencoder for directed acyclic graphs.	Zhang, Jiang, Cui, Garnett, Chen	https://arxiv.org/abs/1904.11088	https://github.com/muhanzhang/D-VAE

Are disentangled representations helpful for abstract reasoning?	van Steenkiste, Locatello, Schmidhuber, Bachem	https://arxiv.org/pdf/1905.12506.pdf

A heuristic for unsupervised model selection for variational disentangled representation learning.	Duan, Watters, Matthey, Burgess, Lerchner, Higgins	https://arxiv.org/pdf/1905.12614.pdf

Dual space learning with variational autoencoders.	Okamoto, Suzuki, Higuchi, Ohsawa, Matsuo	https://pdfs.semanticscholar.org/ea70/6495d4a6214b3d6174bb7fd99c5a9c34c2e6.pdf

Variational autoencoders for sparse and overdispersed discrete data.	Zhao, Rai, Du, Buntine	https://arxiv.org/pdf/1905.00616.pdf

Variational auto-decoder.	Zadeh, Lim, Liang, Morency.	https://arxiv.org/pdf/1903.00840.pdf

Causal discovery with attention-based convolutional neural networks. 	Naura, Bucur, Seifert 	https://www.mdpi.com/2504-4990/1/1/19/pdf

Variational laplace autoencoders.	Park, Kim, Kim	http://proceedings.mlr.press/v97/park19a/park19a.pdf

Variational autoencoders with normalizing flow decoders.		https://openreview.net/forum?id=r1eh30NFwB

Gaussian process priors for view-aware inference.	Hou, Heljakka, Solin	https://arxiv.org/pdf/1912.03249.pdf

SGVAE: sequential graph variational autoencoder.	Jing, Chi, Tang	https://arxiv.org/pdf/1912.07800.pdf	

improving multimodal generative models with disentangled latent partitions.	Daunhawer, Sutter, Vogt	http://bayesiandeeplearning.org/2019/papers/103.pdf	

Cross-population variational autoencoders.	Davison, Severson, Ghosh	https://openreview.net/pdf?id=r1eWdlBFwS http://bayesiandeeplearning.org/2019/papers/96.pdf	

Evidential disambiguation of latent multimodality in conditional variational autoencoders.	Itkina, Ivanovic, Senanayake, Kochenderfer, Pavone	http://bayesiandeeplearning.org/2019/papers/34.pdf	

Increasing the generalisation capacity of conditional VAEs.	Klushyn, Chen, Cseke, Bayer, van der Smagt	https://link.springer.com/chapter/10.1007/978-3-030-30484-3_61	

Multi-source neural variational inference.	Kurle, Gunnemann, van der Smagt	https://wvvw.aaai.org/ojs/index.php/AAAI/article/view/4311	

Early integration for movement modeling in latent spaces.	Hornung, Chen, van der Smagt	https://books.google.co.uk/books?hl=en&lr=&id=M1WfDwAAQBAJ&oi=fnd&pg=PA305&dq=info:MRhvAh4qD7wJ:scholar.google.com&ots=hN84xN5saO&sig=TBMgkFo6z9wrL64TcvzjU4G5gCQ&redir_esc=y#v=onepage&q&f=false
	
Building face recognition system with triplet-based stacked variational denoising autoencoder.	LEe, Hart, Richens, Johri	https://dl.acm.org/citation.cfm?id=3369707	

Cross-domain variational autoencoder for recommender systems.	Shi, Wang	 https://ieeexplore.ieee.org/abstract/document/8935901	
Predictive coding, variational autoencoders, and biological connections.	Marino	https://openreview.net/pdf?id=SyeumQYUUH	

A general and adaptive robust loss function	Barron	https://arxiv.org/pdf/1701.03077.pdf	

Variational autoencoder trajectory primitives and discrete latent. Osa, Ikemoto	https://arxiv.org/pdf/1912.04063.pdf	

Faster attend-infer-repeat with tractable probabilistic models.	Stelzner, Peharz, Kersting	http://proceedings.mlr.press/v97/stelzner19a/stelzner19a.pdf	https://github/stelzner/supair

Learning predictive models from observation and interaction.	Schmeckpeper, Xie, Rybkin, Tian, Daniilidis, Levine, Finn	https://arxiv.org/pdf/1912.12773.pdf

Translating visual art into music	Muller-Eberstein, van Noord	http://openaccess.thecvf.com/content_ICCVW_2019/papers/CVFAD/Muller-Eberstein_Translating_Visual_Art_Into_Music_ICCVW_2019_paper.pdf


## 2018

FFJORD: free-form continuous dynamics for scalable reversible generative models. 	Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud	https://arxiv.org/pdf/1810.01367.pdf

A general method for amortizing variational filtering.	Marino, Cvitkovic, Yue	https://arxiv.org/pdf/1811.05090.pdf	https://github.com/joelouismarino/amortized-variational-filtering

Handling incomplete heterogeneous data using VAEs.	Nazabal, Olmos, Ghahramani, Valera	https://arxiv.org/pdf/1807.03653.pdf	

Sequential attend, infer, repeat: generative modeling of moving objects.	Kosiorek, Kim, Posner, Teh	https://arxiv.org/pdf/1806.01794.pdf	https://github.com/akosiorek/sqair https://www.youtube.com/watch?v=-IUNQgSLE0c&feature=youtu.be

Doubly reparameterized gradient estimators for monte carlo objectives.	Tucker, Lawson, Gu, Maddison	https://arxiv.org/pdf/1810.04152.pdf

Interpretable intuitive physics model. 	Ye, Wang, Davidson, Gupta	https://arxiv.org/pdf/1808.10002.pdf	https://github.com/tianye95/interpretable-intuitive-physics-model

Normalizing Flows Tutorial, Part 2: Modern Normalizing Flows.	Eric Jang	https://blog.evjang.com/2018/01/nf2.html

Neural autoregressive flows.	Huang, Krueger, Lacoste, Courville	https://medium.com/element-ai-research-lab/neural-autoregressive-flows-f164d6b8e462 https://arxiv.org/pdf/1804.00779.pdf	https://github.com/CW-Huang/NAF

Gaussian process prior variational autoencoders.	Casale, Dalca, Sagletti, Listgarten, Fusi	https://papers.nips.cc/paper/8238-gaussian-process-prior-variational-autoencoders.pdf

ACVAE-VC: non-parallel many-to-many voice conversion with auxiliary classifier variational autoencoder.	Kameoka, Kaneko, Tanaka, Hojo	https://arxiv.org/pdf/1808.05092.pdf

Discovering interpretable representations for both deep generative and discriminative models.	Adel, Ghahramani, Weller	http://mlg.eng.cam.ac.uk/adrian/ICML18-Discovering.pdf

Autoregressive quantile networks for generative modelling	. Ostrovski, Dabey, Munos	 https://arxiv.org/pdf/1806.05575.pdf


Probabilistic video generation using holistic attribute control.	He, Lehrmann, Marino, Mori, Sigal	https://arxiv.org/pdf/1803.08085.pdf

Bias and generalization in deep generative models: an empirical study.	Zhao, Ren, Yuan, Song, Goodman, Ermon	https://arxiv.org/pdf/1811.03259.pdf	https://ermongroup.github.io/blog/bias-and-generalization-dgm/ https://github.com/ermongroup/BiasAndGeneralization/tree/master/Evaluate

On variational lower bounds of mutual information.	Poole, Ozair, van den Oord, Alemi, Tucker	http://bayesiandeeplearning.org/2018/papers/136.pdf

GAN - why it is so hard to train generative adversarial networks	. Hui	https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b

Counterfactuals uncover the modular structure of deep generative models. 	Besserve, Sun, Scholkopf. https://arxiv.org/pdf/1812.03253.pdf

Learning independent causal mechanisms.	Parascandolo, Kilbertus, Rojas-Carulla, Scholkopf	https://arxiv.org/pdf/1712.00961.pdf

Emergence of invariance and disentanglement in deep representations.	Achille, Soatto	https://arxiv.org/pdf/1706.01350.pdf

Variational memory encoder-decoder.	Le, Tran, Nguyen, Venkatesh	https://arxiv.org/pdf/1807.09950.pdf	https://github.com/thaihungle/VMED

Variational autoencoders for collaborative filtering.	Liang, Krishnan, Hoffman, Jebara	https://arxiv.org/pdf/1802.05814.pdf

Invariant representations without adversarial training.	Moyer, Gao, Brekelmans, Steeg, Galstyan	http://papers.nips.cc/paper/8122-invariant-representations-without-adversarial-training.pdf	https://github.com/dcmoyer/inv-rep

Density estimation: Variational autoencoders.	Rui Shu	http://ruishu.io/2018/03/14/vae/

TherML: The thermodynamics of machine learning.	Alemi, Fishcer	https://arxiv.org/pdf/1807.04162.pdf

Leveraging the exact likelihood of deep latent variable models.	Mattei, Frellsen	https://arxiv.org/pdf/1802.04826.pdf

What is wrong with VAEs?	Kosiorek	http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html	

Stochastic variational video prediction.	Babaeizadeh, Finn, Erhan, Campbell, Levine	https://arxiv.org/pdf/1710.11252.pdf	https://github.com/alexlee-gk/video_prediction

Variational attention for sequence-to-sequence models.	Bahuleyan, Mou, Vechtomova, Poupart	https://arxiv.org/pdf/1712.08207.pdf	https://github.com/variational-attention/tf-var-attention

FactorVAE Disentangling by factorizing.	Kim, Minh	https://arxiv.org/pdf/1802.05983.pdf	

Disentangling factors of variation with cycle-consistent variational autoencoders.	Jha, Anand, Singh, Veeravasarapu	https://arxiv.org/pdf/1804.10469.pdf	https://github.com/ananyahjha93/cycle-consistent-vae

Isolating sources of disentanglement in VAEs.	Chen, Li, Grosse, Duvenaud	https://arxiv.org/pdf/1802.04942.pdf

VAE with a VampPrior.	Tomczak, Welling	https://arxiv.org/pdf/1705.07120.pdf	

A Framework for the quantitative evaluation of disentangled representations.	Eastwood, Williams	https://openreview.net/pdf?id=By-7dz-AZ	https://github.com/cianeastwood/qedr

Recent advances in autoencoder based representation learning.	Tschannen, Bachem, Lucic	https://arxiv.org/pdf/1812.05069.pdf	

InfoVAE: Balancing learning and inference in variational autoencoders.	Zhao, Song, Ermon	https://arxiv.org/pdf/1706.02262.pdf	

Understanding disentangling in Beta-VAE.	Burgess, Higgins, Pal, Matthey, Watters, Desjardins, Lerchner	https://arxiv.org/pdf/1804.03599.pdf	

Hidden Talents of the Variational autoencoder.	Dai, Wang, Aston, Hua, Wipf	https://arxiv.org/pdf/1706.05148.pdf	

Variational Inference of disentangled latent concepts from unlabeled observations.	Kumar, Sattigeri, Balakrishnan	https://arxiv.org/abs/1711.00848	

Self-supervised learning of a facial attribute embedding from video.	Wiles, Koepke, Zisserman	http://www.robots.ox.ac.uk/~vgg/publications/2018/Wiles18a/wiles18a.pdf	

Wasserstein auto-encoders.	Tolstikhin, Bousquet, Gelly, Scholkopf	https://arxiv.org/pdf/1711.01558.pdf

A two-step disentanglement. method	Hadad, Wolf, Shahar
http://openaccess.thecvf.com/content_cvpr_2018/papers/Hadad_A_Two-Step_Disentanglement_CVPR_2018_paper.pdf	https://github.com/naamahadad/A-Two-Step-Disentanglement-Method

Taming VAEs.	Rezende, Viola	https://arxiv.org/pdf/1810.00597.pdf	https://github.com/denproc/Taming-VAEs https://github.com/syncrostone/Taming-VAEs

IntroVAE Introspective variational autoencoders for photographic image synthesis.	Huang, Li, He, Sun, Tan	https://arxiv.org/pdf/1807.06358.pdf	https://github.com/dragen1860/IntroVAE-Pytorch

Information constraints on auto-encoding variational bayes.	Lopez, Regier, Jordan, Yosef	https://papers.nips.cc/paper/7850-information-constraints-on-auto-encoding-variational-bayes.pdf	https://github.com/romain-lopez/HCV

Learning disentangled joint continuous and discrete representations.	Dupont	https://papers.nips.cc/paper/7351-learning-disentangled-joint-continuous-and-discrete-representations.pdf	https://github.com/Schlumberger/joint-vae

Neural discrete representation learning.	van den Oord, Vinyals, Kavukcuoglu	https://arxiv.org/pdf/1711.00937.pdf	https://github.com/1Konny/VQ-VAE  https://github.com/ritheshkumar95/pytorch-vqvae

Disentangled sequential autoencoder.	Li, Mandt	https://arxiv.org/abs/1803.02991	https://github.com/yatindandi/Disentangled-Sequential-Autoencoder

Variational Inference: A review for statisticians.	Blei, Kucukelbir, McAuliffe	https://arxiv.org/pdf/1601.00670.pdf	
Advances in Variational Inferece.	Zhang, Kjellstrom	https://arxiv.org/pdf/1711.05597.pdf	

Auto-encoding total correlation explanation.	Goa, Brekelmans, Steeg, Galstyan	https://arxiv.org/abs/1802.05822	Closest: https://github.com/gregversteeg/CorEx

Fixing a broken ELBO.	Alemi, Poole, Fischer, Dillon, Saurous, Murphy	https://arxiv.org/pdf/1711.00464.pdf

The information autoencoding family: a lagrangian perspective on latent variable generative models.	Zhao, Song, Ermon	https://arxiv.org/pdf/1806.06514.pdf	https://github.com/ermongroup/lagvae

Debiasing evidence approximations: on importance-weighted autoencoders and jackknife variational inference.	Nowozin	https://openreview.net/pdf?id=HyZoi-WRb	https://github.com/microsoft/jackknife-variational-inference

Unsupervised discrete sentence representation learning for interpretable neural dialog generation.	Zhao, Lee, Eskenazi	https://vimeo.com/285802293 https://arxiv.org/pdf/1804.08069.pdf	https://github.com/snakeztc/NeuralDialog-LAED

Dual swap disentangling.	Feng, Wang, Ke, Zeng, Tao, Song	https://papers.nips.cc/paper/7830-dual-swap-disentangling.pdf	

Multimodal generative models for scalable weakly-supervised learning.	Wu, Goodman	https://papers.nips.cc/paper/7801-multimodal-generative-models-for-scalable-weakly-supervised-learning.pdf	https://github.com/mhw32/multimodal-vae-public https://github.com/panpan2/Multimodal-Variational-Autoencoder

Do deep generative models know what they don't know?	Nalisnick, Matsukawa, The, Gorur, Lakshminarayanan	https://arxiv.org/pdf/1810.09136.pdf	

Glow: generative flow with invertible 1x1 convolutions.	Kingma, Dhariwal	https://arxiv.org/pdf/1807.03039.pdf	https://github.com/openai/glow https://github.com/pytorch/glow

Inference suboptimality in variational autoencoders.	Cremer, Li, Duvenaud	https://arxiv.org/pdf/1801.03558.pdf	https://github.com/chriscremer/Inference-Suboptimality

Adversarial Variational Bayes: unifying variational autoencoders and generative adversarial networks.	Mescheder, Mowozin, Geiger	https://arxiv.org/pdf/1701.04722.pdf	https://github.com/LMescheder/AdversarialVariationalBayes

Semi-amortized variational autoencoders.	Kim, Wiseman, Miller, Sontag, Rush	https://arxiv.org/pdf/1802.02550.pdf	https://github.com/harvardnlp/sa-vae

Spherical Latent Spaces for stable variational autoencoders.	Xu, Durrett	https://arxiv.org/pdf/1808.10805.pdf	https://github.com/jiacheng-xu/vmf_vae_nlp

Hyperspherical variational auto-encoders.	Davidson, Falorsi, De Cao, Kipf, Tomczak	https://arxiv.org/pdf/1804.00891.pdf	https://github.com/nicola-decao/s-vae-tf https://github.com/nicola-decao/s-vae-pytorch

Fader networks: manipulating images by sliding attributes.	Lample, Zeghidour, Usunier, Bordes, Denoyer, Ranzato	https://arxiv.org/pdf/1706.00409.pdf	https://github.com/facebookresearch/FaderNetworks

Training VAEs under structured residuals.	Dorta, Vicente, Agapito, Campbell, Prince, Simpson	https://arxiv.org/pdf/1804.01050.pdf	https://github.com/Garoe/tf_mvg

oi-VAE: output interpretable VAEs for nonlinear group factor analysis.	Ainsworth, Foti, Lee, Fox	https://arxiv.org/pdf/1802.06765.pdf	https://github.com/samuela/oi-vae

infoCatVAE: representation learning with categorical variational autoencoders.	Lelarge, Pineau	https://arxiv.org/pdf/1806.08240.pdf	https://github.com/edouardpineau/infoCatVAE

Iterative Amortized inference.	Marino, Yue, Mandt	https://arxiv.org/pdf/1807.09356.pdf https://vimeo.com/287766880	https://github.com/joelouismarino/iterative_inference

On unifying Deep Generative Models.	Hu, Yang, Salakhutdinov, Xing	https://arxiv.org/pdf/1706.00550.pdf	

Diverse Image-to-image translation via disentangled representations.	Lee, Tseng, Huang, Singh, Yang	https://arxiv.org/pdf/1808.00948.pdf	https://github.com/HsinYingLee/DRIT

PIONEER networks: progressively growing generative autoencoder.	Heljakka, Solin, Kannala	https://arxiv.org/pdf/1807.03026.pdf	https://github.com/AaltoVision/pioneer

Towards a definition of disentangled representations.	Higgins, Amos, Pfau, Racaniere, Matthey, Rezende, Lerchner	https://arxiv.org/pdf/1812.02230.pdf	

Life-long disentangled representation learning with cross-domain latent homologies.	Achille, Eccles, Matthey, Burgess, Watters, Lerchner,  Higgins	file:///Users/matthewvowels/Downloads/Life-Long_Disentangled_Representation_Learning_wit.pdf	

Learning deep disentangled embeddings with F-statistic loss.	Ridgeway, Mozer	https://arxiv.org/pdf/1802.05312.pdf	https://github.com/kridgeway/f-statistic-loss-nips-2018

Learning latent subspaces in variational  autoencoders.	Klys, Snell, Zemel	https://arxiv.org/pdf/1812.06190.pdf

On the latent space of Wasserstein auto-encoders. Rubenstein, Scholkopf, Tolstikhin.	https://arxiv.org/pdf/1802.03761.pdf	https://github.com/tolstikhin/wae

Learning disentangled representations with Wasserstein auto-encoders.	Rubenstein, Scholkopf, Tolstikhin	https://openreview.net/pdf?id=Hy79-UJPM	

The mutual autoencoder: controlling information in latent code representations.	Phuong, Kushman, Nowozin, Tomioka, Welling	https://openreview.net/pdf?id=HkbmWqxCZ   https://openreview.net/pdf?id=HkbmWqxCZ http://2017.ds3-datascience-polytechnique.fr/wp-content/uploads/2017/08/DS3_posterID_048.pdf	

Auxiliary guided autoregressive variational autoencoders.	Lucas, Verkbeek	https://openreview.net/pdf?id=HkGcX--0-	https://github.com/pclucas14/aux-vae

Interventional robustness of deep latent variable models.	Suter, Miladinovic, Bauer, Scholkopf	https://pdfs.semanticscholar.org/8028/a56d6f9d2179416d86837b447c6310bd371d.pdf?_ga=2.190184363.1450484303.1564569882-397935340.1548854421	

Understanding degeneracies and ambiguities in attribute transfer.	Szabo, Hu, Portenier, Zwicker, Facaro	http://openaccess.thecvf.com/content_ECCV_2018/papers/Attila_Szabo_Understanding_Degeneracies_and_ECCV_2018_paper.pdf	
DNA-GAN: learning disentangled representations from multi-attribute images.
Xiao, Hong, Ma	https://arxiv.org/pdf/1711.05415.pdf	https://github.com/Prinsphield/DNA-GAN

Normalizing flows.	Kosiorek	http://akosiorek.github.io/ml/2018/04/03/norm_flows.html	

Hamiltonian variational auto-encoder	Caterini, Doucet, Sejdinovic	https://arxiv.org/pdf/1805.11328.pdf	

Causal generative neural networks.	Goudet, Kalainathan, Caillou, Guyon, Lopez-Paz, Sebag.	https://arxiv.org/pdf/1711.08936.pdf	https://github.com/GoudetOlivier/CGNN

Flow-GAN: Combining maximum likelihood and adversarial learning in generative models.	Grover, Dhar, Ermon	https://arxiv.org/pdf/1705.08868.pdf	https://github.com/ermongroup/flow-gan

Linked causal variational autoencoder for inferring paired spillover effects.	Rakesh, Guo, Moraffah, Agarwal, Liu	https://arxiv.org/pdf/1808.03333.pdf	https://github.com/rguo12/CIKM18-LCVA

Unsupervised anomaly detection via variational auto-encoder for seasonal KPIs in web applications. 	Xu, Chen, Zhao, Li, Bu, Li, Liu, Zhao, Pei, Feng, Chen, Wang, Qiao	https://arxiv.org/pdf/1802.03903.pdf

Mutual information neural estimation.	Belghazi, Baratin, Rajeswar, Ozair, Bengio, Hjelm.	https://arxiv.org/pdf/1801.04062.pdf	https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation- https://github.com/mzgubic/MINE

Explorations in homeomorphic variational auto-encoding.	Falorsi, de Haan, Davidson, Cao, Weiler, Forre, Cohen.	https://arxiv.org/pdf/1807.04689.pdf	https://github.com/pimdh/lie-vae

Hierarchical variational memory network for dialogue generation.	Chen, Ren, Tang, Zhao, Yin	http://delivery.acm.org/10.1145/3190000/3186077/p1653-chen.pdf?ip=86.162.136.199&id=3186077&acc=OPEN&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1569938843_c07ad21d173fc64a44a22fd6521140cb

World models.	Ha, Schmidhuber	https://arxiv.org/pdf/1803.10122.pdf

## 2017

Opening the black box of deep neural networks via information. 	Schwartz-Ziv, Tishby	https://arxiv.org/pdf/1703.00810.pdf	https://www.youtube.com/watch?v=gOn8Po_NPe4


Discovering causal signals in images	. Lopez-Paz, Nishihara, Chintala, Scholkopf, Bottou 	https://arxiv.org/pdf/1605.08179.pdf

Autoencoding variational inference for topic models.	Srivastava, Sutton	https://arxiv.org/pdf/1703.01488.pdf

Hidden Markov model variational autoencoder for acoustic unit discovery. 	Ebbers, Heymann, Drude, Glarner, Haeb-Umbach, Raj	https://www.isca-speech.org/archive/Interspeech_2017/pdfs/1160.PDF

Application of variational autoencoders for aircraft turbomachinery design. 	Zalger	http://cs229.stanford.edu/proj2017/final-reports/5231979.pdf

Semi-supervised learning with variational autoencoders.	Keng	 http://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/

Causal effect inference with deep latent variable models.	Louizos, Shalit, Mooij, Sontag, Zemel, Welling	https://arxiv.org/pdf/1705.08821.pdf	https://github.com/AMLab-Amsterdam/CEVAE

beta-VAE: learning basic visual concepts with a constrained variational framework.	Higgins, Matthey, Pal, Burgess, Glorot, Botvinick, Mohamed, Lerchner	https://openreview.net/pdf?id=Sy2fzU9gl	

Challenges in disentangling independent factors of variation.	Szabo, Hu, Portenier, Facaro, Zwicker	https://arxiv.org/pdf/1711.02245.pdf	https://github.com/ananyahjha93/challenges-in-disentangling

Composing graphical models with neural networks for structured representations and fast inference.	Johnson, Duvenaud, Wiltschko, Datta, Adams	https://arxiv.org/pdf/1603.06277.pdf	

Split-brain autoencoders: unsupervised learning by cross-channel prediction.	Zhang, Isola, Efros	https://arxiv.org/pdf/1611.09842.pdf	

Learning disentangled representations with semi-supervised deep generative models.Siddharth, Paige, van de Meent, Desmaison, Goodman, Kohli, Wood, Torr	https://papers.nips.cc/paper/7174-learning-disentangled-representations-with-semi-supervised-deep-generative-models.pdf	https://github.com/probtorch/probtorch

Learning hierarchical features from generative models.	Zhao, Song, Ermon	https://arxiv.org/pdf/1702.08396.pdf	https://github.com/ermongroup/Variational-Ladder-Autoencoder

Multi-level variational autoencoder: learning disentangled representations from grouped observations.	Bouchacourt, Tomioka, Nowozin	https://arxiv.org/pdf/1705.08841.pdf	

Neural Face editing with intrinsic image disentangling.	Shu, Yumer, Hadap, Sankavalli, Shechtman, Samaras	http://openaccess.thecvf.com/content_cvpr_2017/papers/Shu_Neural_Face_Editing_CVPR_2017_paper.pdf	https://github.com/zhixinshu/NeuralFaceEditing

Variational Lossy Autoencoder.	Chen, Kingma, Salimans, Duan, Dhariwal, Schulman, Sutskever, Abbeel	https://arxiv.org/abs/1611.02731	https://github.com/jiamings/tsong.me/blob/master/_posts/reading/2016-11-08-lossy-vae.md

Unsupervised learning of disentangled and interpretable representations from sequential data.	Hsu, Zhang, Glass	https://papers.nips.cc/paper/6784-unsupervised-learning-of-disentangled-and-interpretable-representations-from-sequential-data.pdf	https://github.com/wnhsu/FactorizedHierarchicalVAE https://github.com/wnhsu/ScalableFHVAE

Factorized variational autoencoder for modeling audience reactions to movies.	Deng, Navarathna, Carr, Mandt, Yue, Matthews, Mori	http://www.yisongyue.com/publications/cvpr2017_fvae.pdf	

Learning latent representations for speech generation and transformation.	Hsu, Zhang, Glass	https://arxiv.org/pdf/1704.04222.pdf	https://github.com/wnhsu/SpeechVAE

Unsupervised learning of disentangled representations from video.	Denton, Birodkar	https://papers.nips.cc/paper/7028-unsupervised-learning-of-disentangled-representations-from-video.pdf	https://github.com/ap229997/DRNET

Laplacian pyramid of conditional variational autoencoders.	Dorta, Vicente, Agapito, Campbell, Prince, Simpson	http://cs.bath.ac.uk/~nc537/papers/cvmp17_LapCVAE.pdf	

Neural Photo Editing with Inrospective Adverarial Networks.	Brock, Lim, Ritchie, Weston	https://arxiv.org/pdf/1609.07093.pdf	https://github.com/ajbrock/Neural-Photo-Editor

Discrete Variational Autoencoder.	Rolfe	https://arxiv.org/pdf/1609.02200.pdf	https://github.com/QuadrantAI/dvae

Reinterpreting importance-weighted autoencoders.	Cremer, Morris, Duvenaud	https://arxiv.org/pdf/1704.02916.pdf	https://github.com/FighterLYL/iwae

Density Estimation using realNVP.	Dinh, Sohl-Dickstein, Bengio	https://arxiv.org/pdf/1605.08803.pdf	https://github.com/taesungp/real-nvp https://github.com/chrischute/real-nvp

JADE: Joint autoencoders for disentanglement.	Banijamali, Karimi, Wong, Ghosi	https://arxiv.org/pdf/1711.09163.pdf	
Joint Multimodal learning with deep generative models.	Suzuki, Nakayama, Matsuo	https://openreview.net/pdf?id=BkL7bONFe	https://github.com/masa-su/jmvae

Towards a deeper understanding of variational autoencoding models.	Zhao, Song, Ermon	https://arxiv.org/pdf/1702.08658.pdf	https://github.com/ermongroup/Sequential-Variational-Autoencoder

Lagging inference networks and posterior collapse in variational autoencoders.	Dilokthanakul, Mediano, Garnelo, Lee, Salimbeni, Arulkumaran, Shanahan	https://arxiv.org/pdf/1611.02648.pdf	https://github.com/Nat-D/GMVAE https://github.com/psanch21/VAE-GMVAE

On the challenges of learning with inference networks on sparse, high-dimensional data.	Krishnan, Liang, Hoffman	https://arxiv.org/pdf/1710.06085.pdf	https://github.com/rahulk90/vae_sparse

Stick-breaking Variational Autoencoder.		https://arxiv.org/pdf/1605.06197.pdf	https://github.com/sporsho/hdp-vae

Deep variational canonical correlation analysis.	Wang, Yan, Lee, Livescu	https://arxiv.org/pdf/1610.03454.pdf	https://github.com/edchengg/VCCA_pytorch

Nonparametric variational auto-encoders for hierarchical representation learning.	Goyal, Hu, Liang, Wang, Xing	https://arxiv.org/pdf/1703.07027.pdf	https://github.com/bobchennan/VAE_NBP/blob/master/report.markdown

PixelSNAIL: An improved autoregressive generative model.	Chen, Mishra, Rohaninejad, Abbeel	https://arxiv.org/pdf/1712.09763.pdf	https://github.com/neocxi/pixelsnail-public

Improved Variational Inference with inverse autoregressive flows.	Kingma, Salimans, Jozefowicz, Chen, Sutskever, Welling	https://arxiv.org/pdf/1606.04934.pdf	https://github.com/kefirski/bdir_vae

It takes (only) two: adversarial generator-encoder networks.	Ulyanov, Vedaldi, Lempitsky	https://arxiv.org/pdf/1704.02304.pdf	https://github.com/DmitryUlyanov/AGE

Symmetric Variational Autoencoder and connections to adversarial learning.	Chen, Dai, Pu, Li, Su, Carin	https://arxiv.org/pdf/1709.01846.pdf	

Reconstruction-based disentanglement for pose-invariant face recognition.	Peng, Yu, Sohn, Metaxas, Chandraker	https://arxiv.org/pdf/1702.03041.pdf	https://github.com/zhangjunh/DR-GAN-by-pytorch

Is maximum likelihood useful for representation learning?	Huszár	https://www.inference.vc/maximum-likelihood-for-representation-learning-2/	

Disentangled representation learning GAN for pose-invariant face recognition.	Tran, Yin, Liu	http://zpascal.net/cvpr2017/Tran_Disentangled_Representation_Learning_CVPR_2017_paper.pdf	https://github.com/kayamin/DR-GAN

Improved Variational Autoencoders for text modeling using dilated convolutions.	Yang, Hu, Salakhutdinov, Berg-kirkpatrick	https://arxiv.org/pdf/1702.08139.pdf	

Improving variational auto-encoders using householder flow.	Tomczak, Welling	https://arxiv.org/pdf/1611.09630.pdf	https://github.com/jmtomczak/vae_householder_flow

Sticking the landing: simple, lower-variance gradient estimators for variational inference. Roeder, Wu, Duvenaud.	http://proceedings.mlr.press/v97/kingma19a/kingma19a.pdf	https://github.com/geoffroeder/iwae

VEEGAN: Reducing mode collapse in GANs using implicit variational learning.	Srivastava, Valkov, Russell, Gutmann.	https://arxiv.org/pdf/1705.07761.pdf	https://github.com/akashgit/VEEGAN

Discovering discrete latent topics with neural variational inference.	Miao, Grefenstette, Blunsom	https://arxiv.org/pdf/1706.00359.pdf

Variational approaches for auto-encoding generative adversarial networks. 	Rosca, Lakshminarayana, Warde-Farley, Mohamed	https://arxiv.org/pdf/1706.04987.pdf

Variational Autoencoder and extensions.	Courville	https://ift6266h17.files.wordpress.com/2017/03/vae1.pdf

A neural representation of sketch drawings.	Ha, Eck	https://arxiv.org/pdf/1704.03477.pdf

## 2016

Attend, infer, repeat: fast scene understanding with generative models.	Eslami, Heess, Weber, Tassa, Szepesvari, Kavukcuoglu, Hinton	https://arxiv.org/pdf/1603.08575.pdf	http://akosiorek.github.io/ml/2017/09/03/implementing-air.html https://github.com/aleju/papers/blob/master/neural-nets/Attend_Infer_Repeat.md

Deep feature consistent variational autoencoder.	Hou, Shen, Sun, Qiu	https://arxiv.org/pdf/1610.00291.pdf	https://github.com/sbavon/Deep-Feature-Consistent-Variational-AutoEncoder-in-Tensorflow

Neural variational inference for text processing. 	Miao, Yu, Grefenstette,  Blunsom.	https://arxiv.org/pdf/1511.06038.pdf

Domain-adversarial training of neural networks.	Ganin, Ustinova, Ajakan, Germain, Larochelle, Laviolette, Marchand, Lempitsky	https://arxiv.org/pdf/1505.07818.pdf

Tutorial on Variational Autoencoders.	Doersch	https://arxiv.org/pdf/1606.05908.pdf

How to train deep variational autoencoders and probabilistic ladder networks.	Sonderby, Raiko, Maaloe, Sonderby, Winther	https://orbit.dtu.dk/files/121765928/1602.02282.pdf	

ELBO surgery: yet another way to carve up the variational evidence lower bound.	Hoffman, Johnson	http://approximateinference.org/accepted/HoffmanJohnson2016.pdf	

Variational inference with normalizing flows.	Rezende, Mohamed	https://arxiv.org/pdf/1505.05770.pdf	

The Variational Fair Autoencoder.	Louizos, Swersky, Li, Welling, Zemel	https://arxiv.org/pdf/1511.00830.pdf	https://github.com/dendisuhubdy/vfae

Information dropout: learning optimal representations through noisy computations.	Achille, Soatto	https://arxiv.org/pdf/1611.01353.pdf	

Domain separation networks.	Bousmalis, Trigeorgis, Silberman, Krishnan, Erhan	https://arxiv.org/pdf/1608.06019.pdf	https://github.com/fungtion/DSN  https://github.com/farnazj/Domain-Separation-Networks

Disentangling factors of variation in deep representations using adversarial training.	Mathieu, Zhao, Sprechmann, Ramesh, LeCunn	https://arxiv.org/pdf/1611.03383.pdf	https://github.com/ananyahjha93/disentangling-factors-of-variation-using-adversarial-training

Variational autoencoder for semi-supervised text classification.	Xu, Sun, Deng, Tan	https://arxiv.org/pdf/1603.02514.pdf	https://github.com/wead-hsu/ssvae related: https://github.com/isohrab/semi-supervised-text-classification

Learning what and where to draw.	Reed, Sohn, Zhang, Lee	https://arxiv.org/pdf/1610.02454.pdf	

Attribute2Image: Conditional image generation from visual attributes.	Yan, Yang, Sohn, Lee	https://arxiv.org/pdf/1512.00570.pdf	

Variational inference with normalizing flows.	Rezende, Mohamed	https://arxiv.org/pdf/1505.05770.pdf	https://github.com/ex4sperans/variational-inference-with-normalizing-flows

Wild Variational Approximations.	Li, Liu	http://approximateinference.org/2016/accepted/LiLiu2016.pdf	

Importance Weighted Autoencoders.	Burda,  Grosse, Salakhutdinov	https://arxiv.org/pdf/1509.00519.pdf	https://github.com/yburda/iwae https://github.com/xqding/Importance_Weighted_Autoencoders
https://github.com/abdulfatir/IWAE-tensorflow

Stacked What-Where Auto-encoders.	Zhao, Mathieu, Goroshin, LeCunn	https://arxiv.org/pdf/1506.02351.pdf	https://github.com/yselivonchyk/Tensorflow_WhatWhereAutoencoder

Disentangling nonlinear perceptual embeddings with multi-query triplet networks.	Veit, Belongie, Karaletsos	https://www.researchgate.net/profile/Andreas_Veit/publication/301837223_Disentangling_Nonlinear_Perceptual_Embeddings_With_Multi-Query_Triplet_Networks/links/57e2997308ae040ae3c2f3a3/Disentangling-Nonlinear-Perceptual-Embeddings-With-Multi-Query-Triplet-Networks.pdf	

Ladder Variational Autoencoders.	Sonderby, Raiko, Maaloe, Sonderby, Winther	https://arxiv.org/pdf/1602.02282.pdf	
Variational autoencoder for deep learning of images, labels and captions.	Pu, Gan Henao, Yuan, Li, Stevens, Carin	https://papers.nips.cc/paper/6528-variational-autoencoder-for-deep-learning-of-images-labels-and-captions.pdf	

Approximate inference for deep latent Gaussian mixtures.	Nalisnick, Hertel, Smyth	https://pdfs.semanticscholar.org/f6fe/5e8e25994c188ba6a124462e2cc55f2c5a67.pdf	https://github.com/enalisnick/mixture_density_VAEs

Auxiliary Deep Generative Models.	Maaloe, Sonderby, Sonderby, Winther	https://arxiv.org/pdf/1602.05473.pdf	https://github.com/larsmaaloee/auxiliary-deep-generative-models

Variational methods for conditional multimodal deep learning.	Pandey, Dukkipati	https://arxiv.org/pdf/1603.01801.pdf	

PixelVAE: a latent variable model for natural images.	Gulrajani, Kumar, Ahmed, Taiga, Visin, Vazquez, Courville	https://arxiv.org/pdf/1611.05013.pdf	https://github.com/igul222/PixelVAE https://github.com/kundan2510/pixelVAE

Adversarial autoencoders.	Makhzani, Shlens, Jaitly, Goodfellow, Frey	https://arxiv.org/pdf/1511.05644.pdf	https://github.com/conan7882/adversarial-autoencoders

A hierarchical latent variable encoder-decoder model for generating dialogues.	Serban, Sordoni, Lowe, Charlin, Pineau, Courville, Bengio	http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf	

Infinite variational autoencoder for semi-supervised learning.	Abbasnejad, Dick	https://arxiv.org/pdf/1611.07800.pdf	

f-GAN: Training generative neural samplers using variational divergence minimization.	Nowozin, Cseke	https://arxiv.org/pdf/1606.00709.pdf	https://github.com/LynnHo/f-GAN-Tensorflow

DISCO Nets: DISsimilarity Coefficient networks	Bouchacourt, Kumar, Nowozin	https://arxiv.org/pdf/1606.02556.pdf	https://github.com/oval-group/DISCONets

Information dropout: learning optimal representations through noisy computations.	Achille, Soatto	https://arxiv.org/pdf/1611.01353.pdf

Weakly-supervised disentangling with recurrent transformations for 3D view synthesis.	Yang, Reed, Yang, Lee	https://arxiv.org/pdf/1601.00706.pdf	https://github.com/jimeiyang/deepRotator

Autoencoding beyond pixels using a learned similarity metric.	Boesen, Larsen, Sonderby, Larochelle, Winther	https://arxiv.org/pdf/1512.09300.pdf	https://github.com/andersbll/autoencoding_beyond_pixels

Generating images with perceptual similarity metrics based on deep networks	Dosovitskiy, Brox.	https://arxiv.org/pdf/1602.02644.pdf	https://github.com/shijx12/DeepSim

A note on the evaluation of generative models.	Theis, van den Oord, Bethge.	https://arxiv.org/pdf/1511.01844.pdf

InfoGAN: interpretable representation learning by information maximizing generative adversarial nets.	Chen, Duan, Houthooft, Schulman, Sutskever, Abbeel	https://arxiv.org/pdf/1606.03657.pdf	https://github.com/openai/InfoGAN

Disentangled representations in neural models. 	Whitney	https://arxiv.org/abs/1602.02383

A recurrent latent variable model for sequential data.	Chung, Kastner, Dinh, Goel, Courville, Bengio	https://arxiv.org/pdf/1506.02216.pdf

Unsupervised learning of 3D structure from images.	Rezende, Eslami, Mohamed, Battaglia, Jaderberg, Heess	https://arxiv.org/pdf/1607.00662.pdf


## 2015
Deep learning and the information bottleneck principle	Tishby, Zaslavsky	https://arxiv.org/pdf/1503.02406.pdf

Training generative neural networks via Maximum Mean Discrepancy optimization.	Dziugaite, Roy, Ghahramani	https://arxiv.org/pdf/1505.03906.pdf


NICE: non-linear independent components estimation.	Dinh, Krueger, Bengio	https://arxiv.org/pdf/1410.8516.pdf	

Deep convolutional inverse graphics network.	Kulkarni, Whitney, Kohli, Tenenbaum	https://arxiv.org/pdf/1503.03167.pdf	https://github.com/yselivonchyk/TensorFlow_DCIGN

Learning structured output representation using deep conditional generative models.	Sohn, Yan, Lee	https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models.pdf	https://github.com/wsjeon/ConditionalVariationalAutoencoder

Latent variable model with diversity-inducing mutual angular regularization.	Xie, Deng, Xing	https://arxiv.org/pdf/1512.07336.pdf

DRAW: a recurrent neural network for image generation.	Gregor, Danihelka, Graves, Rezende, Wierstra.	https://arxiv.org/pdf/1502.04623.pdf	https://github.com/ericjang/draw

Variational Inference II.	Xing, Zheng, Hu, Deng	https://www.cs.cmu.edu/~epxing/Class/10708-15/notes/10708_scribe_lecture13.pdf	

## 2014

Auto-encoding variational Bayes.	Kingma, Welling	https://arxiv.org/pdf/1312.6114.pdf

Learning to disentangle factors of variation with manifold interaction.	Reed, Sohn, Zhang, Lee	http://proceedings.mlr.press/v32/reed14.pdf	

Semi-supervised learning with deep generative models.	Kingma, Rezende, Mohamed, Welling	https://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models.pdf	https://github.com/saemundsson/semisupervised_vae https://github.com/Response777/Semi-supervised-VAE

Stochastic backpropagation and approximate inference in deep generative models.	Rezende, Mohamed, Wierstra	https://arxiv.org/pdf/1401.4082.pdf	https://github.com/ashwindcruz/dgm/tree/master/adgm_mnist

Representation learning: a review and new perspectives.	Bengio, Courville, Vincent	https://arxiv.org/pdf/1206.5538.pdf

## 2011
Transforming Auto-encoders.	Hinton, Krizhevsky, Wang	https://www.cs.toronto.edu/~hinton/absps/transauto6.pdf	


## 2008

Graphical models, exponential families, and variational inference.	Wainwright, Jordan et al

## 2004

Variational learning and bits-back coding: an information-theoretic view to Bayesian learning.	Honkela, Valpola	https://www.cs.helsinki.fi/u/ahonkela/papers/infview.pdf	

## 2000

The information bottleneck method.	Tishby, Pereira, Bialek	https://arxiv.org/pdf/physics/0004057.pdf
