---
layout: post
title:  "Dense codes"
date:   2016-02-12 11:33:45 +0100
comments: true
---

![Publish your negative results cartoon.](/assets/dense/negative-data.png)

Hello.

My name is Rasmus. I’ve worked at Tradeshift as a software developer for roughly 4 years. Among other things I’ve helped build the machine learning engine behind the CloudScan product. Cloudscan takes unstructured PDF invoices and turns them into a structured XML format.

I’ve always wanted to do a PhD in machine learning as I believe it’s our best approach towards artificial intelligence. Fortunately Tradeshift shares my interest in machine learning, and they agreed to support my research ambitions.

After a sizeable amount of paperwork I’m happy to report the project was approved and I started the PhD program in November 2015.

The overall project goal is to improve the digitization of invoices using deep learning, and hopefully learn a thing or two along the way.

Over the next 3 years I’ll be blogging about my progress. I’ll try to write as simple as possible without dumbing it down. I’ll try to be honest about the ups and downs, u-turns and dead ends.

As much as I’d love to publish some ground breaking results that will not be the case for this first blog post. Today I’ll publish some negative results on an idea I had. Publishing negative or inconclusive results are important - the results might be the missing piece in someone else’s puzzle, it could prevent others from duplicating the same experiments or even better, someone might be inspired to carry the idea forward.

## The idea

[Categorical variables](https://en.wikipedia.org/wiki/Categorical_variable) occur in many machine learning problems. When encoding categorical variables for use in neural networks, one typically use [one-hot](https://en.wikipedia.org/wiki/One-hot) encoding (Bishop 2006). The advantages of this encoding is that it’s simple and that it makes no assumptions about the similarity of the values of the variable, since all values will have the same distance to every other value: a [hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) of 2, or an [euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) of $$\sqrt{2}$$. One disadvantage is that the length of the vector of the encoded values grows linearly with the number of unique values. For [neural nets](https://en.wikipedia.org/wiki/Artificial_neural_network) this means the number of parameters of the model also grows linearly with the number of unique values. For text where the values are unique words or phrases, this vector can become very big. [Google’s Word2Vec model](https://code.google.com/archive/p/word2vec/) trained on 3 million words and phrases results in a massive 3.4 gigabytes of parameters. That’s a lot of numbers!

The idea we had is simple. Replace the one-hot encoding with a dense binary encoding. A dense binary code of N bits can represent $$2^N$$ unique values. With just 32 bits you can encode ~4.3 billion different values. In other words, to represent $$Q$$ unique values with a one-hot encoding, you’d need a vector with $$Q$$ elements, whereas you’d only need $$log_2(Q)$$ elements if you were to use a binary encoding. See below for a comparison of the two encodings.

<table class="pure-table pure-table-bordered">
    <thead>
    <tr>
        <th>word</th>
        <th>one-hot</th>
        <th>binary</th>
    </tr>
    </thead>
    <tr>
        <td>man</td>
        <td>00000001</td>
        <td>000</td>
    </tr>
    <tr>
        <td>dog</td>
        <td>00000010</td>
        <td>001</td>
    </tr>
    <tr>
        <td>cat</td>
        <td>00000100</td>
        <td>010</td>
    </tr>
    <tr>
        <td>car</td>
        <td>00001000</td>
        <td>100</td>
    </tr>
    <tr>
        <td>table</td>
        <td>00010000</td>
        <td>011</td>
    </tr>
    <tr>
        <td>lamp</td>
        <td>00100000</td>
        <td>101</td>
    </tr>
    <tr>
        <td>fish</td>
        <td>01000000</td>
        <td>110</td>
    </tr>
    <tr>
        <td>woman</td>
        <td>10000000</td>
        <td>111</td>
    </tr>
</table>

One disadvantage of the dense binary encoding is that the values are not all the same distance from each other. In the example above ‘man’ is closer to ‘dog’, ‘cat’ and ‘car’, than to the other classes, and furthest from ‘woman’. This distance is artificial, and in the given example, actually counter intuitive.

So why do we think this encoding might work? Well, if you assign the binary codes randomly then the number of bits that are not equal between two values will follow the [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution) with $$p=0.5$$. Imagine you draw two random binary vectors. The bits in each of the two vectors are randomly 1 or 0. Now look at the first bit in the two vectors. There are 4 equally likely outcomes: 11, 00, 10, 01. In half of the outcomes the bits are equal and in the other half of the outcomes they’re not equal. Since all the bits in the vectors are independent, this is the binomial distribution with $$p=0.5$$.

The mean number of different bits is $$np$$ and the standard deviation is $$\sqrt{np(1-p)}$$. So as the number of bits grows the distribution becomes more centered around the mean, which is half the number of bits used. Also, with more bits the probability of two values having zero different bits, i.e. a collision, quickly becomes astronomically small even for very large amount of unique values. You can test it yourself [here](http://everydayinternetstuff.com/2015/04/hash-collision-probability-calculator/).

Further we reasoned that even though some of the values will be closer to each other than others and every value will share its bits with roughly half the other values, this messy kind of data is exactly what machine learning is built to handle. Take [MNIST](http://yann.lecun.com/exdb/mnist/) for an example. The inputs are the pixel intensity values between 0 and 1 in a 28 by 28 image of a handwritten digit. The output is which digit it was, from 0 to 9. One pixel in the input will be turned on for many different digits, of different classes. The task of the model is to disentangle this, and combine the many different pixels, which are each a weak clue, into one coherent label.

As such we reasoned that any complexity that we added to the task by use of the dense encoding, could be mitigated by simply adding more power to the model. In the case of neural networks in the form of more, or bigger layers.

One-hot encoded categorical values appear both in the input and in the output. We made experiments with the dense encoding on both sides. We’ll cover the input side first.

## Input side

### Toy data set

Before beginning any real work we tested out the idea on a toy data set. The data set is the simplest possible. There is a single categorical feature, with K values, and there are K classes, each corresponding to a value of the categorical feature. e.g. if the categorical variable is ‘A’ then the class is 1, ‘B’ equals 2, etc.

We used a [multinomial logistic regression](http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/) model. It’s essentially a neural network with no hidden layers. We set K=1024 and created 10000 samples randomly. First we trained a control model with the feature being one-hot encoded. Not surprisingly the accuracy was 100%.

For each of the 1024 feature values we created a random 40 bit binary code, and trained an otherwise identical model with this binary encoding of the feature. We chose 40 bit as it give a very low ($$<10^{-6}$$) probability of a collision. The accuracy was 100% again. So we went from 1024 bits in the input, and a model with $$1024^2$$ parameters, to a model with 40 bits in the input and $$40\cdot1024$$, or roughly 4% of the parameters, with no loss of accuracy. For 16 bit codes the accuracy is around 99%. The code is just 37 lines and can be found [here](https://gist.github.com/rasmusbergpalm/1b99d65e0e8b81a99013).

This was encouraging so we proceeded testing the idea on a more realistic data set.

### Mimicking Word2Vec

We chose to use the Word2Vec (Mikolov et al. 2013) paradigm to showcase our dense binary encoding. This was chosen because we believed there was a possibility of drastically reducing the number of parameters in the model. If we could reduce the 3.4 gigabytes to a couple of megabytes that was sure to raise some eyebrows.

The purpose of the Word2Vec models are to learn a so called ‘embedding’ of words or phrases. This embedding is a vector of real numbers for each word or phrase. The vector is typically between 100 to 300 numbers. The embedding is good if words or phrases that are semantically or syntactically similar, are close (e.g. euclidean, [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)). In the example above we’d expect ‘man’ and ‘woman’ to be closer than ‘car’ and ‘fish’. In a good embedding you can also do arithmetic on the embedded words, e.g. embedding(‘king’) - embedding(‘man’) + embedding(‘woman’) ~ embedding(‘queen’). You can try it yourself [here](http://deeplearner.fz-qqq.net/).

It’s a very nice result and the embeddings often form the basis for more advanced models such as those in machine translation.

The simplest formulation of the Word2Vec model is a standard neural network, in which the input is a one-hot encoded word, there’s a single hidden layer, called the embedding layer, and the output is another one-hot encoded word. The model is trained to predict which words might appear in the context of a given word. The size of the context, is called the window size, and determines how many words to the left and right of a word is considered its context. For instance, given the sentence ‘the quick fox’, and a window size of 1, then the following pairs of words would be the [input, output] words the model was trained on: [the, quick], [quick, the], [quick, fox], [fox, quick]. Data is readily available as no manual labeling is required. For instance you can download the entire english wikipedia and use every article as training data to get a good embedding of English words.

![Word2Vec model.](/assets/dense/w2v.png)
*Figure 1. A simple Word2Vec model. The boxes are layers in a neural net. Following [Andrej Karpathy's](http://karpathy.github.io/) convention the red layer is the input layer, the green layers are the hidden layers and the blue layer is the output layer.*

Despite the relatively simple description above, implementing a full Word2Vec model is not trivial as there are a number of tricks to speed it up, as well as just handling all the data, etc. So instead of learning a full Word2Vec model, we first attempted to learn the embedding of the pre-trained model trained on 3 million words and phrases directly. See Figure 2. The intuition was that we’d need to learn a similar embedding implicitly with a full model anyway, so we might as well see if we could learn it explicitly with a model that was simpler to implement. It also might give us some hints as to which kind of architectures was sufficient to learn this embedding.

![Mimicking Word2Vec model.](/assets/dense/w2v-fit.png)
*Figure 2. Learning the embedding directly from a pre-trained Word2Vec model, with a dense binary code as input.*

This idea is called ‘model compression’ (Buciluǎ, Caruana, and Niculescu-Mizil 2006). It’s quite simple; you train a simpler model to mimic a more complex model. It’s been used to learn shallow neural nets from deep neural nets (Ba and Caruana 2013) and learn a single model from an [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) of models (Hinton, Vinyals, and Dean 2015).

Why do we think that we can compress the number of parameters in the first place? Well, the whole point of Word2Vec is that the embedding of similar words is close. Take for instance the words ‘Man’ and ‘man’ as an extreme example. The only difference being the capitalization we’d expect these words to be very close in the embedding space, at least in the dimensions that capture the semantic meaning of a word, meaning they’ll have nearly the same parameters for these dimensions. This redundancy should be compressible.

So, what we’re trying to learn is a function from our dense binary input code of a word, to the 300 dimensional embedding of that word. We trained three neural nets: 4 layers with 300, 600 and 1200 units respectively. We used the [rectified linear unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) and batch normalization (Ioffe and Szegedy 2015) in all the hidden layers. The output layer was linear. We normalised the embedding, such that for each dimension the mean was 0 and the standard deviation was 1. Although this changes the embedding, and probably makes it worse (since the distances between words will be changed), we can always recover the original embedding. We trained the network to minimize [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE). We used a 256 bit binary code calculated using the SHA256 [hash](https://en.wikipedia.org/wiki/Hash_function) of the word. The chance of a collision for 256 bits and 3 million entries are approximately $$4\cdot10^{-65}$$, so we reckoned it should be enough bits.

Initially we trained on all 3 million words and phrases. These models converged to a loss of approximately 1. A model that always outputs zero for all 300 dimensions would get a loss of 1. This can be seen from looking at the definition of the mean squared error.

$$MSE(x, y) = \frac{1}{N} \sum_i^N \sqrt{(f(x_i)-y_i)^2}$$

If the model output $$f(x)$$ is equal to the mean of $$y$$ then the MSE is identical to the variance of $$y$$. Since we normalized $$y$$ the mean is 0 and the variance is 1. If the model always outputs 0 then it’s equal to the mean of $$y$$, and thus the MSE is equal to the variance, namely 1.

This meant that the models were failing to learn. As a sanity check we trained on only 1 word, to see if the model could learn that. We got a loss of approximately zero, so yes, indeed it could. We then increased the amount of words and phrases we trained on from 256 to 131072, to see when the model started failing.

We used a batch size as big as the number of words and phrases we trained on, i.e. we did gradient descent, not stochastic gradient descent. We trained the models for 2000 epochs equaling 2000 parameter updates, after which we they had converged. This was assessed by letting a few models run for 10000 parameter updates and plotting the progress, which did not change much after 2000 updates. We did not worry about [overfitting](https://en.wikipedia.org/wiki/Overfitting) since we did not expect the model to generalize to unseen words in the first place. We did try applying [dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) to the hidden layers and training for up to 30000 epochs for one model but this gave a much worse loss (~0.42 to ~0.77). The results can be seen in Figure 3.

![Mimicking Word2Vec results.](/assets/dense/w2v-fit-results.png)
*Figure 3. Mimicking pre-trained Word2Vec embeddings using a dense binary code as input. The number of unique words and phrases were varied and the network sizes were kept constant.*

So, it appears that at some point the model capacity is saturated, and as we include more words and phrases it gets worse. The really interesting question though, is whether we managed any compression at all. If we can perfectly mimic the embedding of a number of words and phrases with fewer parameters than the original Word2Vec model, then it’s just a matter of adding enough parameters to the model until we can perfectly mimic all 3 million, and we’d have achieved some level of compression. To investigate we plotted the loss as a function of the amount of parameters of our models relative to how many parameters the original model had for a given amount of words and phrases. See figure 4.

![Mimicking Word2Vec results.](/assets/dense/w2v-fit-results2.png)
*Figure 4. Mimicking pre-trained Word2Vec embeddings using a dense binary code as input. Relative parameters are relative to how many parameters the original model being mimicked used.*

From figure 4 we can see that we only get close to zero loss as the relative parameters approaches 1. So it appears that we failed to achieve any compression. We cannot represent the same function with fewer parameters than the original by using a dense binary code and a more powerful network.

What we can say is that we can mimic the original model with a deeper network, approximately the same amount of parameters and the dense binary code. It’s not really a groundbreaking result, but it mirrors the results of (Ba and Caruana 2013) nicely. In it they show that the reverse is true: Given a deep network you can mimic it with a shallow network with approximately the same amount of parameters.

We did not experiment with how good these mimicked embeddings were in terms of the word analogy task, e.g. man is to woman as king is to queen, etc. It would be interesting to see how the performance degraded as a function of the MSE from the original embedding.

I don’t understand why we could not achieve any level of compression. I still think the intuition behind the compressibility of the ‘Man’, ‘man’ case is sound. I have a bad feeling that there’s some fundamental information theoretic or linear algebra argument we’ve failed to realize we’re up against.

### Full Word2Vec

Despite the negative results on compressing a pre-trained Word2Vec model, we decided to experiment with a full model. The reasoning was that maybe it would be possible for the model to learn a good embedding ‘under its own terms’. In the previous experiments we forced the model to mimic a pre-trained embedding. This pre-trained embedding might not have been suitable with the dense binary encoding, or in other words, there might be another embedding, equally good, but more compatible with the dense binary encoding.

First we reproduced the completely standard Word2Vec model results. We trained a model on the first [100 million bytes](http://mattmahoney.net/dc/text8.zip) of the English wikipedia, and measured accuracy on the word analogy task as described above, using the [question-words data set](https://word2vec.googlecode.com/svn/trunk/questions-words.txt). We used a window size of 5, e.g. 2 words left and right, and used [negative sampling](http://stackoverflow.com/questions/27860652/word2vec-negative-sampling-in-layman-term) to speed up the training. We discarded words that did not occur at least 5 times, and we did not subsample frequent words. This gave us a dictionary of size 71290 unique words. We used a batch size of 1000 words, and an embedding size of 300. The results can be seen in the table below.

<table class="pure-table pure-table-bordered">
    <thead>
    <tr>
        <th>epoch</th>
        <th>loss</th>
        <th>accuracy</th>
    </tr>
    </thead>
    <tr>
        <td>1</td>
        <td>1.445437</td>
        <td>24.7%</td>
    </tr>
    <tr>
        <td>2</td>
        <td>1.346631</td>
        <td>32.8%</td>
    </tr>
</table>

After two epochs, which took a couple of hours, we manually stopped the training. Having determined that the code and model worked well enough, and having established an unoptimized baseline we ran an experiment with the dense codes. In case the dense codes were better than the baseline we could always go back and improve the baseline.

The model and training regime was equal to the standard Word2Vec model except where otherwise noted. We again used the SHA256 hashing function to get a 256 bit dense binary code. With the previous results in mind we opted for the most powerful model of 4 hidden layers with 1200 units, plus a linear hidden embedding layer of 300 units. The 4x1200 hidden layers used batch normalization and the rectified linear activation function. See figure 5. This model has 62% of the parameters compared to the standard Word2Vec model, and 23% if you only count the parameters up to the embedding layer, i.e. if you disregard the embedding-to-output parameters in common for both models.

![Full Word2Vec model.](/assets/dense/w2v-full.png)
*Figure 5. The full Word2Vec model trained with the dense binary input codes.*

We also trained a model with 4 layers of 2628 units. This model has almost exactly the same amount of parameters as the original Word2Vec model. We let both models train over a weekend or more than 20 epochs. The best results, measured on accuracy, can be seen below.

<table class="pure-table pure-table-bordered">
    <thead>
    <tr>
        <th>model</th>
        <th>epoch</th>
        <th>loss</th>
        <th>accuracy</th>
    </tr>
    </thead>
    <tr>
        <td>4 x 1200</td>
        <td>23</td>
        <td>1.270343</td>
        <td>23.6%</td>
    </tr>
    <tr>
        <td>4 x 2628</td>
        <td>20</td>
        <td>1.253298</td>
        <td>28.8%</td>
    </tr>
</table>

So, after more than 10 times the epochs the results are not as good as for the standard, unoptimized Word2Vec model, even with the same number of parameters in the model.

Curiously the loss is better, while the accuracy is worse. In other words the embedding is better at predicting the context words, but worse in terms of the word analogy task. This runs counter to my intuition that the two should be very closely correlated. I don’t have any good explanation for this.

It seems that you can use the dense binary encoding, but you’ll have to use more complex (deeper, harder to optimize) models with approximately as many parameters and train them for longer in order to get slightly worse results than the equivalent one-hot encoded model. The results on the toy data set contradicts this statement, but I suspect the toy data set was too simple. More experiments are necessary, to figure out when the dense binary code works as well as the one-hot encoding. If this boundary is found it might be possible to build some intuition as to why it works under these circumstances.

The experiments we performed with the Word2Vec models are somewhat strange. In one case we’re trying to mimic a model, and in the other we’re optimizing one thing (predicting context words), but measuring the model performance on a different task (word analogy). It would be interesting to perform some experiments with the dense codes on a more standard problem formulation, e.g. a classification problem with categorical input variables, where model performance was measured on the classification task itself.

## Output side

[Multi-class classification](https://en.wikipedia.org/wiki/Multiclass_classification) problems, in which you are asked to classify an input into one of more than two classes, are ubiquitous. The typical neural net approach to these problems is to use a [softmax](https://en.wikipedia.org/wiki/Softmax_function) output layer with one-hot encoded classes. As the number of classes grows the computational complexity of a softmax output layer grows linearly. This can make training multi-class models with many classes quite slow. This is especially true for language modelling in which there can be thousands or even millions of unique words and phrases.

If it was possible to instead represent the classes as a dense binary code and derive a [loss function](https://en.wikipedia.org/wiki/Loss_function) with a computational complexity depending on the number of bits of the code instead of the number of classes one could drastically reduce the computational burden for multi-class problems with a large amount of classes.

Ole Winther, my supervisor, suggested the following loss function

$$L(x, y') = -\log\left(\frac{\prod_i e^{f(x)_iy'_i}} {\sum_y \prod_i e^{f(x)_iy_i}}\right)$$

Here, $$x$$ is the input to the network and $$y'$$ is the encoded target class. $$f(x)$$ is the output of a linear layer. $$i$$ iterates over the positions in the output vector and the sum over $$y$$ iterates over all the encoded classes.

This loss function has some nice properties. First of all, it’s a generalization of the normal softmax loss function. If you use the one-hot encoding for the classes $$y$$, the normal softmax function falls out. Secondly, when using the dense binary codes, it reduces to a much simpler expression that does not depend on iterating over all possible permutations of the output, but rather depend only linearly on the number of elements in the output:

$$L(x, y') = \sum_i [\log(1+e^{f(x)_i})-f(x)_iy'_i]$$

Interestingly, this is exactly equal to the negative log likelihood of a multi-label problem with sigmoid output units, where the targets are 0 or 1.

At test time we chose the encoded class with the largest cosine similarity to the output the model produced. More specifically. If $$Y$$ contains the binary encoded classes, one per row, and $$r$$ iterates over the rows, we do the following:

$$
\newcommand\norm[1]{\left\lVert#1\right\rVert}
\DeclareMathOperator*{\argmax}{arg\,max}
U = 2Y-1 \\
\hat{y} = \argmax_r \frac{f(x)U_r}{\norm{f(x)}\norm{U_r}}
$$

The idea of reducing the computational complexity by using a binary code has been proposed before, under the name hierarchical softmax (Morin and Bengio 2005; Mnih and Hinton 2009). There’s a very nice explanation of the idea [here](https://www.youtube.com/watch?v=B95LTf2rVWM). In this formulation the amount of parameters still increase linearly with the amount of classes, while the computational complexity at training time is reduced to log the amount of classes. Also, they use binary codes such that each bit represent a meaningful separation of the classes. This approach has been shown to perform well on language modelling tasks.

In our approach we attempt to reduce both the computational complexity and the number of parameters to log the amount of classes, and we use completely random binary codes.

Using random dense binary codes for output have also been examined before under the term ‘error correcting output codes’ (Dietterich and Bakiri 1995; Bautista et al. 2012). This approach was mainly motivated by an intuition that if the class was represented with more than the needed bits, the extra bits could be used for error correcting, e.g. the model might get a couple of bits wrong but the correct class could still be recovered.

Our approach differs from these since we’re using a new loss function, and comparing our approach to modern multi-layer neural networks.

### Concatenated MNIST

MNIST is a very common machine learning benchmark. It consists of 70.000 black and white 28 by 28 images of handwritten digits from 0 through 9. 10 classes was not enough so we concatenated MNIST images together and made the resulting class the resulting number. This way we could easily control the number of classes by simply concatenating more images together.

![Sample of five concatenated MNIST digits.](/assets/dense/mnist.png)
*Figure 6. Example of 5 concatenated MNIST digits. The class for this input would be '65723'.*

We first verified our code by training a neural network on the normal MNIST, e.g. with 10 classes, until we achieved close to state-of-the-art for normal neural networks (i.e. not [convolutional](https://en.wikipedia.org/wiki/Convolutional_neural_network)). We got an error rate of 0.91% with 3 hidden layers of 1200 units, using leaky rectified linear units, batch normalization, a softmax output layer and dropout of 0.5 on the hidden layers and 0.2 on the input. Once the code had been shown to work, we performed the experiments on the concatenated MNIST.

We concatenated five digits, resulting in 100.000 classes. The training set in the original MNIST is 50.000 digits (60.000 really, but we used 10.000 for validation) which means there are $$50.000^5$$ way of combining those digits for the concatenated MNIST. Instead of training on all $$50.000^5$$ permutations every epoch we instead sampled 50.000 random permutations every epoch. We did the same for the validation and test set, whenever we needed to validate or test.

We used a random binary code length of 256 and we trained five models with 3 hidden layers of 1200, 2400, 3600, 4800 and 6000 units respectively. See figure 7. In order to do a fair comparison between the dense binary code and the one-hot encoding we trained 3 layer models with one-hot encoded output, with the number of units in the hidden layers chosen so that the total number of parameters of the models were as close as possible, namely 76, 206, 391, 629 and 918 units.

![Model for the concatenated MNIST](/assets/dense/mnist-model.png)
*Figure 7. The architecture used for the dense binary code MNIST experiments. We experimented with W values of 1200, 2400, 3600, 4800 and 6000*

All hidden layers used batch normalization and leaky rectifier units. Models were trained until their validation accuracy had not improved for 1000 epochs or to a maximum of 2000 epochs. Once a model finished training, the accuracy on the test set was evaluated at the epoch with the best validation accuracy. We did not use any dropout. We did try, but the results were worse for the dense coding models and the same for the one-hot models and for both cases the models took longer to converge. Dropout was most likely not needed due to the amount of unique examples. The results can be seen in Figure 8.

![Results for the concatenated MNIST](/assets/dense/mnist-results.png)
*Figure 8. Results on 5 concatenated MNIST digits for the dense random binary coding and the one-hot coding.*

For the small networks (1200 vs. 76 and 2400 vs. 206), the one-hot coding is best. For the bigger networks, the dense codes are slightly better. The data can be seen below.

<table class="pure-table pure-table-bordered">
    <thead>
    <tr>
        <th>dense network</th>
        <th>one-hot network</th>
        <th>dense accuracy</th>
        <th>one-hot accuracy</th>
    </tr>
    </thead>
    <tr>
        <td>3x1200</td>
        <td>3x76</td>
        <td>0.4913</td>
        <td>0.73</td>
    </tr>
    <tr>
        <td>3x2400</td>
        <td>3x206</td>
        <td>0.8137</td>
        <td>0.8351</td>
    </tr>
    <tr>
        <td>3x3600</td>
        <td>3x391</td>
        <td>0.8832</td>
        <td>0.8793</td>
    </tr>
    <tr>
        <td>3x4800</td>
        <td>3x629</td>
        <td>0.9035</td>
        <td>0.8963</td>
    </tr>
    <tr>
        <td>3x6000</td>
        <td>3x907</td>
        <td>0.9121</td>
        <td>0.9046</td>
    </tr>
</table>

While the accuracy is somewhat better for the larger networks, this was not the kind of result we were looking for. We expected to be able to drastically reduce the amount of parameters and computational complexity by using the dense codes. The results instead show that when we reduce the amount of parameters, then the one-hot encoding is actually better than a dense model with the same number of parameters. The improved accuracy using the dense codes for the bigger models is an interesting result, and further experiments should be made to figure out under which circumstances these results hold.

## Conclusion

We hypothesized that using a dense random binary encoding of categorical variables could lead to dramatic reductions in the number of parameters and computational complexity when training neural networks.

Our results did not support the hypothesis. We performed experiments, both on the input and the output side. The models using dense random binary coding needed approximately as many parameters as the ones using one-hot encoding in order to achieve similar performance, and was somewhat harder to train.

The one exception to this observed pattern was the experiment on the toy data set, in which we could losslessly compress 1024 categorical values to 40 bits. We believe that the toy data set was too simple, but would encourage more experiments to find the boundary when the binary encoding starts to fail.

## Personal learnings

The experiments and results described here are the tip of the iceberg. Many more experiments were run, and later discarded which have caused me to alternate between believing dense coding worked extremely well, and not at all. It also means I’ve wasted a lot of time. I’ve listed the three main reasons for discarding results below and my thoughts on how to avoid it in the future.

### Bugs...

This is always a problem when you’re coding, but it is especially grievous when you’re experimenting with machine learning models. Bugs in normal code will cause an unexpected behavior or simply cause the program to crash. You normally protect yourself from this by writing unit tests.

The bugs causing crashes are easy enough to fix in the machine learning experiments as well. It’s the unexpected behavior ones that are devious. When doing an experiment, you don’t know what to expect, that’s why you do the experiment in the first place. Sure, you might have an intuition, but you don’t *know*, and, being a good scientist, you should be able to accept any outcome.

Further many bugs will simply cause the performance to be slightly worse than optimal, not outright bad, since any working parts of the model will attempt to correct any errors produced by the bad parts of the model during training. The problem is you don’t know whether you have a bug, your data is not well suited to the model, your hyper parameter tuning is bad or that your idea is simply just bad. How do you separate out these causes of poor performance in a reliable and fast way?

As an example, for about a week I got 0% accuracy on the word analogy task for my full Word2Vec models using the dense binary code. I had implemented a normal Word2Vec model with the same data and almost the same code, and the loss was decreasing as I would expect. I looked through my code several times but could not find anything wrong. Finally I accepted that the embedding was just not very good at the word analogy task. It was only later, almost by accident, that I discovered that I had forgotten a ```deterministic=True``` in my test function. This was not a problem in the original Word2Vec model since it was not using batch normalization.

Maybe you think it’s just a stupid mistake that you’d never make. You might be right. But it’s human to fail. It’s what we do. We make assumptions, we’re [biased](https://en.wikipedia.org/wiki/List_of_cognitive_biases) and we take shortcuts. We can’t pay attention to everything. The programming community has accepted this a long time ago. Static analysis, unit tests and code review are pillars of any serious software company. I don’t like to think about how many good ideas may have been deemed a failure and left unpublished because of a simple bug.

I can’t think of any form of automated tests that could catch errors like this. If you have any ideas for a way to discover these kinds of error, preferably in an automated way, please write in the comments below. My only idea is to get my fellow PhD students to perform code review. It also has some other nice side effects such as knowledge sharing. I’ll see if I can get them on to the idea.

### Not training for long enough

I deemed many experiments as failures simply because I did not let them train for long enough. As an example I trained the one-hot encoded models for the concatenated MNIST first, and the validation accuracy improved after just a few epochs. After a couple of hundred epochs they were converged. I stopped the models after they had not improved for more than 100 epochs which was more than enough patience.

When I trained the models with the dense encoding, I kept the 100 epoch patience, and all the models got 0% validation accuracy. Figure 9 shows the first 400 epochs of a typical experiment when training on the concatenated MNIST with dense binary codes. The accuracy and loss plateau is more than 100 epochs wide, which caused the models to stop training before they had converged.

![Loss plateau when training on concatenated MNIST with dense codes](/assets/dense/loss-plateau.png)
*Figure 9. Loss and accuracy plateau when training on concatenated MNIST with dense binary codes.*

How do I know when to stop training then? How do I know I’m not just stuck in another plateau with my current results? I guess the short answer is I don’t. Looking at previous results on the same data with a comparable model and extrapolating from that seems to be the best you can do. But even though that’s just what I did I still fell into this trap.

Hopefully my intuition for this will become better over the years but for now I guess I’ll just have to train my models for far longer than I’d normally assume they’d need just to be on the safe side. If you have any good tips or tricks on how to know when it’s safe to stop training (assuming you're not overfitting), please leave a comment.

### Not comparing apples to apples

I got results that pointed in all directions until I started to compare models that had the same amount of parameters. It’s not really surprising that a model with more parameters than another is better. When comparing results you should take the number of parameters and training time into account.

## References

 * Ba, Lei Jimmy, and Rich Caruana. 2013. “Do Deep Nets Really Need to Be Deep?” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1312.6184.
 * Bautista, Miguel Ángel, Sergio Escalera, Xavier Baró, Petia Radeva, Jordi Vitriá, and Oriol Pujol. 2012. “Minimal Design of Error-Correcting Output Codes.” Pattern Recognition Letters 33 (6): 693–702.
 * Bishop, Christopher M. 2006. “Pattern Recognition.” Machine Learning. academia.edu. http://www.academia.edu/download/30428242/bg0137.pdf.
 * Buciluǎ, Cristian, Rich Caruana, and Alexandru Niculescu-Mizil. 2006. “Model Compression.” In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 535–41. ACM.
 * Dietterich, T. G., and G. Bakiri. 1995. “Solving Multiclass Learning Problems via Error-Correcting Output Codes.” The Journal of Artificial Intelligence Research. jair.org. http://www.jair.org/papers/paper105.html.
 * Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. 2015. “Distilling the Knowledge in a Neural Network.” arXiv [stat.ML]. arXiv. http://arxiv.org/abs/1503.02531.
 * Ioffe, Sergey, and Christian Szegedy. 2015. “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.” arXiv [cs.LG]. arXiv. http://arxiv.org/abs/1502.03167.
 * Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. 2013. “Distributed Representations of Words and Phrases and Their Compositionality.” In Advances in Neural Information Processing Systems 26, edited by C. J. C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Q. Weinberger, 3111–19. Curran Associates, Inc.
 * Mnih, Andriy, and Geoffrey E. Hinton. 2009. “A Scalable Hierarchical Distributed Language Model.” In Advances in Neural Information Processing Systems 21, edited by D. Koller, D. Schuurmans, Y. Bengio, and L. Bottou, 1081–88. Curran Associates, Inc.
 * Morin, Frederic, and Yoshua Bengio. 2005. “Hierarchical Probabilistic Neural Network Language Model.” In Aistats, 5:246–52. Citeseer.