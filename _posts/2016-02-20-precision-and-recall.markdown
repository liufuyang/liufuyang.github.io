---
layout: post
title:  "Understanding Precision and Recall"
date:   2016-02-20
comments: true
---

$$precision = \frac{true \space  positive} {no. \space  of \space predicted \space positive }$$

$$recall = \frac {true \space positive} {no. \space of \space actual \space positive}$$

If we increase the prediction bar from 0.5 to higher, this will predict y=1 only when we are very confident.

* High precision, lower recall -> confident about positive; (Precise on predicting positive, but you predict just very)

If we want to avoid **false negatives** (for example, avoid missing too many case of cancer), we can decrease prediction threshold lower than 0.5.

* Low precision, high recall -> avoid false negatives; (Not all precise but we cover more actual positive)

## How do we compare precision and recall values?

Suppose we now have different algorithms and using them giving different precision and recall values, how do we judge which model has the best performance?

<table class="pure-table pure-table-bordered">
    <thead>
    <tr>
        <th></th>
        <th>Precision (P)</th>
        <th>Recall (R)</th>
    </tr>
    </thead>
    <tr>
        <td>Algorithm 1 </td>
        <td>0.5</td>
        <td>0.4</td>
    </tr>
    <tr>
        <td>Algorithm 2</td>
        <td>0.7</td>
        <td>0.1</td>
    </tr>
    <tr>
        <td>Algorithm 3</td>
        <td>0.02</td>
        <td>1.0</td>
    </tr>
</table>

We need a single compare metric.

## $$F_1$$ score:

$$F_1 = 2 \frac{PR}{P+R}$$

Then simply run validation with different value of threshold to get a model with the highest $$F_1$$ value.
