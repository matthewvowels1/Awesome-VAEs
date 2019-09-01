# Awesome-VAEs
Awesome work on the VAE, disentanglement, representation learning, and generative models. 

I gathered these resources (currently @ 208 papers) as literature for my PhD, and thought it may come in useful for others. This list includes works relevant to various topics relating to VAEs. Sometimes this spills over to topics e.g. adversarial training and GANs, general disentanglement, variational inference, flow-based models and auto-regressive models. Always keen to expand the list.  I have also included an excel file which includes notes on each paper, as well as a breakdown of the topics covered in each paper.

They are ordered by year (new to old). I provide a link to the paper as well as to the github repo where available.

## 2019

Reweighted expectation maximization.	Dieng, Paisley	https://arxiv.org/pdf/1906.05850.pdf	https://github.com/adjidieng/REM

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


## 2018

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

## 2017

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


## 2016

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

2016	InfoGAN: interpretable representation learning by information maximizing generative adversarial nets.	Chen, Duan, Houthooft, Schulman, Sutskever, Abbeel	https://arxiv.org/pdf/1606.03657.pdf	https://github.com/openai/InfoGAN


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

## 2000

The information bottleneck method.	Tishby, Pereira, Bialek	https://arxiv.org/pdf/physics/0004057.pdf
