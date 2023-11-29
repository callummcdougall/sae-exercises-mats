This is a GitHub repo for hosting some exercises in sparse autoencoders, which I've recently finished working on as part of the upcoming ARENA 3.0 iteration. Having spoken to Neel Nanda and others in interpretability-related MATS streams, it seemed useful to make these exercises accessible out of the context of the rest of the ARENA curriculum.

### **Links to Colabs:** [**Exercises**](https://colab.research.google.com/drive/1rPy82rL3iZzy2_Rd3F82RwFhlVnnroIh?usp=sharing)**,** [**Solutions**](https://colab.research.google.com/drive/1fg1kCFsG0FCyaK4d5ejEsot4mOVhsIFH?usp=sharing)**.**

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/galaxies.jpeg" width="300">

If you don't like working in Colabs, then you can clone this repo, download the exercises & solutions Colabs as notebooks, and run them in the same directory.

The exercises were built out from the Toy Models of Superposition exercises from the previous iteration, but now with new Sparse Autoencoder content. These exercises fall into 2 groups:

## **SAEs in toy models**

We take the toy models from Anthropic's [Toy Models of Superposition paper](https://transformer-circuits.pub/2022/toy_model/index.html) (which there are also exercises for), and train sparse autoencoders on the representations learned by these toy models. These exercises culminate in using **neuron resampling** to successfully recover all the learned features from the toy model of bottleneck superposition - see [this animation](https://github.com/callummcdougall/sae-exercises-mats/blob/main/animation_2.gif).

## **SAEs in real models**

And there are exercises on interpreting an SAE trained on a transformer, where you can discover some [cool learned features](https://www.perfectlynormal.co.uk/blog-sae) (e.g. a neuron exhibiting skip trigam-like behaviour, which activates on left-brackets following Django-related sytax, and predicts the completion `('` -\> `django`).

You can either read through the Solutions colab (which has all output displayed & explained), or go through the Exercises colab and fill in the functions according to the specifications you are given, looking at the Solutions when you're stuck. Both colabs come with test functions you can run to verify your solution works.

## **List of all exercises**

I've listed all the exercises down here, along with prerequisites (although I expect most readers will only be interested in the sparse autoencoder exercises). Each set of exercises  is labelled with their prerequisites. For instance, the label (1*, 3) means the first set of exercises is essential, and the third is recommended but not essential.

Abbreviations: TMS = Toy Models of Superposition, SAE = Sparse Autoencoders.

1.  **TMS: Superposition in a Nonprivileged Basis**
2.  **TMS: Correlated / Anticorrelated Features** (1*)
3.  **TMS: Superposition in a Privileged Basis** (1*)
4.  **TMS: Feature Geometry** (1*)
5.  **SAEs in Toy Models** (1*, 3)
6.  **SAEs in Real Models** (1*, 5*, 3)

---

Please reach out to me if you have any questions or suggestions about these exercises (either by email at `cal.s.mcdougall@gmail.com`, or a LessWrong private message / comment on this post). Happy coding!
