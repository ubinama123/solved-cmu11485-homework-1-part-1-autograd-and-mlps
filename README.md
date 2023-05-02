Download Link: https://assignmentchef.com/product/solved-cmu11485-homework-1-part-1-autograd-and-mlps
<br>
<ul>

 <li><strong>Overview:</strong>

  <ul>

   <li><strong>MyTorch</strong>: An explanation of the library structure, Autograd, the local autograder, and how to submit to Autolab. You can get the starter code from Autolab or the course website.</li>

   <li><strong>Autograd</strong>: Part 1 one will cumulate in a functioning Autograd library, which will be autograded on Autolab. This is worth 40 points.</li>

   <li><strong>MLPs</strong>: Part 2 will be to use and expand your Autograd to complete an MLP library with Linear, ReLU, and BatchNorm layers. It will also have cross entropy loss and an SGD optimizer with momentum. All of these will be autograded. This is worth 60 points.</li>

   <li><strong>MNIST</strong>: Part 3 will be to use your MLP library to train on the MNIST dataset. This is worth</li>

  </ul></li>

</ul>

<ul>

 <li><strong>Appendix</strong>: This contains information that you may (read: will) find useful.</li>

</ul>

<ul>

 <li><strong>Directions:</strong>

  <ul>

   <li>We estimate this assignment may take up to 15 hours, with much of that time digesting concepts. So <strong>start early</strong>, and <strong>come to office hours if you get stuck </strong>to ensure things go as smoothly as possible.</li>

   <li>We recommend that you look through all of the problems before attempting the first problem. However we do recommend you complete the problems in order, as questions often rely on the completion of previous questions.</li>

   <li>You are required to do this assignment using Python3. Other than for testing, do not use any auto-differentiation toolboxes (PyTorch, TensorFlow, Keras, etc) – you are only permitted and recommended to vectorize your computation using the Numpy library.</li>

   <li>Note that Autolab uses <a href="https://numpy.org/doc/1.18/">numpy v1.18.1</a></li>

  </ul></li>

</ul>

<h1>Introduction</h1>

Starting from this assignment, you will be developing your own version of the popular deep learning library PyTorch. It will (cleverly) be called <strong>“MyTorch”</strong>.

A key part of MyTorch will be your implementation of <strong>Autograd</strong>, which is a library for <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">Automatic Differentiation</a> <a href="https://en.wikipedia.org/wiki/Automatic_differentiation">. </a>This feature is new to this semester – previously, students manually programmed in symbolic derivatives for each module. While Autograd may seem hard at first, the code is short and it will save a lot of future time and effort. We’re hoping that you’ll see its value in reducing your workload and also enhancing your understanding of DL in practice.

<strong>Your goal for this assignment is to develop all the components necessary to train an MLP on the </strong><a href="https://en.wikipedia.org/wiki/MNIST_database"><strong>MNIST dataset</strong></a> <a href="https://en.wikipedia.org/wiki/MNIST_database">.</a>

Training on MNIST is considered to the print(“Hello world!”) of DL. But this metaphor usually doesn’t assume you’ve implemented autograd from scratch : )

<h2>Homework Structure</h2>

handout autograder…………………………………………….Files for scoring your code locally hw1 autograder runner.py………………………………………..Files for running autograder tests test mlp.py test mnist.py test autograd.py

data……………………………………………….[Question 3] MNIST Data Files mytorch………………………………………………………………MyTorch library nn…………………………………………………………..Neural Net-related files activations.py batchnorm.py functional.py linear.py loss.py module.py sequential.py

optim………………………………………………………..Optimizer-related files optimizer.py sgd.py

autograd engine.py……………………………………………..Autograd main code tensor.py……………………………………………………………Tensor object sandbox.py…………………….Simple environment to test operations and autograd functions create tarball.sh……………………………….Script for generating Autolab submission grade.sh……………………………………………..Script for running local autograder hw1 mnist.py……………………………………….[Question 3] Running MLP on MNIST

<strong>Note: a prior background in Object-Oriented Programming (OOP) will be helpful, but is not required</strong>. If you’re having difficulty navigating the code, please Google, post on Piazza, or come to TA hours. Remember – you are building this library yourself. Mistakes early on can lead to a cascade of problems later,

<table width="625">

 <tbody>

  <tr>

   <td width="24">and</td>

   <td colspan="3" width="27"><strong>will</strong></td>

   <td width="17">be</td>

   <td colspan="2" width="17">dif</td>

   <td colspan="2" width="7">fi</td>

   <td width="24">cult</td>

   <td colspan="2" width="20">for</td>

   <td colspan="2" width="21">oth</td>

   <td width="18">ers</td>

   <td width="15">to</td>

   <td width="15">de</td>

   <td width="27">bug.</td>

   <td width="35">Make</td>

   <td width="27">sure</td>

   <td width="15">to</td>

   <td width="29">read</td>

   <td width="25">this</td>

   <td width="22">doc</td>

   <td width="7">u</td>

   <td width="31">ment</td>

   <td width="25">care</td>

   <td width="30">fully,</td>

   <td width="22">as</td>

   <td width="8">it</td>

   <td width="23">will</td>

   <td width="13">in</td>

   <td width="15">flu</td>

   <td width="27">ence</td>

   <td width="13">fu</td>

   <td width="24">ture</td>

  </tr>

  <tr>

   <td colspan="2" width="24">com</td>

   <td width="14">po</td>

   <td colspan="3" width="33">nents</td>

   <td colspan="2" width="15">of</td>

   <td colspan="3" width="30">your</td>

   <td colspan="2" width="34">work.</td>

   <td colspan="23" width="474"></td>

  </tr>

  <tr>

   <td width="24"></td>

   <td width="1"></td>

   <td width="19"></td>

   <td width="11"></td>

   <td width="19"></td>

   <td width="4"></td>

   <td width="16"></td>

   <td width="2"></td>

   <td width="6"></td>

   <td width="26"></td>

   <td width="1"></td>

   <td width="21"></td>

   <td width="16"></td>

   <td width="6"></td>

   <td width="20"></td>

   <td width="17"></td>

   <td width="18"></td>

   <td width="26"></td>

   <td width="36"></td>

   <td width="31"></td>

   <td width="17"></td>

   <td width="31"></td>

   <td width="27"></td>

   <td width="24"></td>

   <td width="7"></td>

   <td width="32"></td>

   <td width="28"></td>

   <td width="30"></td>

   <td width="21"></td>

   <td width="10"></td>

   <td width="23"></td>

   <td width="15"></td>

   <td width="15"></td>

   <td width="28"></td>

   <td width="15"></td>

   <td width="24"></td>

  </tr>

 </tbody>

</table>

<h3>0.1       Autograd: An Introduction</h3>

<table width="485">

 <tbody>

  <tr>

   <td width="45"><strong>Please</strong></td>

   <td width="40"><strong>make</strong></td>

   <td width="33"><strong>sure</strong></td>

   <td width="29"><strong>you</strong></td>

   <td width="20"><strong>un</strong></td>

   <td width="22"><strong>der</strong></td>

   <td width="39"><strong>stand</strong></td>

   <td width="23"><strong>Au</strong></td>

   <td width="14"><strong>to</strong></td>

   <td width="33"><strong>grad</strong></td>

   <td width="19"><strong>be</strong></td>

   <td width="28"><strong>fore</strong></td>

   <td width="16"><strong>at</strong></td>

   <td width="40"><strong>tempt</strong></td>

   <td width="23"><strong>ing</strong></td>

   <td width="27"><strong>the</strong></td>

   <td width="33"><strong>code</strong></td>

  </tr>

 </tbody>

</table>

. If you read from now until the start of the first question, you should be good to go.

Also, see <strong>Recitation 2 (Calculating Derivatives)</strong>, which gives you more background and additional support in completing the homework. We’ll release the slides early, with HW1.

<strong>0.1.1        Why autograd?</strong>

Autograd is a framework for automatic differentiation. It’s used to automatically calculate the derivative (or for us, gradient) of <strong>any </strong>computable function<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. This includes NNs, which are just big, big functions.

It turns out that <strong>backpropagation is actually a special case of automatic differentiation</strong>. They both do the same thing: calculate partial derivatives.

This equivalence is very convenient for us. Without autograd, you’d have to manually solve, implement, and re-implement derivative calculations for every individual network component (last year’s homework). This also means that changing any part/setting of that component would require adding more special cases or possibly even re-implementing big chunks of the code.

But with autograd, you only need to implement derivatives for simple operations and functions, which by homework 2, you’ll see will already be good enough to run most common DL components. The rest of the code will be typical <strong>Object-Oriented Programing (OOP)</strong>. Compared to last year’s homework, this is much more flexible, easier to debug, and (eventually) less confusing. <strong>0.1.2 Context: Training an NN</strong>

Above is a typical training routine for an NN. Review it carefully, as it’ll be important context for both explaining autograd and completing the assignment.

<strong>0.1.3           Context: Loss, Weights, and Derivatives</strong>

Important question: what are backprop and autograd doing? Why are they calculating gradients, and what are the gradients used for?

Recall these two calculus concepts:

<ol>

 <li>A partial derivative measures how much the output <em>y </em>would change if we only change the input variable <em>x<sub>i</sub></em>. We assume the other variables are held constant.</li>

 <li>The derivative can be interpreted as the slope of a function at some input point. If the slope is positive and you increase <em>x<sub>i</sub></em>, you should expect to move ‘upward’ (<em>y </em>should increase). Similarly, if the slope is negative, you should expect to move ‘downward’ (<em>y </em>should decrease).</li>

</ol>

In the forward pass, the final loss value depends on the input values, the network params, and the label value. But we’re not interested in the input/label; what we’re interested in is <strong>how exactly each individual network param affected the loss</strong>.

The partial derivatives (or gradients; gradients are just the derivative transposed) that we are calculating in backprop track these relationships: <strong>assuming the other params are held constant, how much would an increase in this param change the loss? </strong>If we increase the param’s value and the loss increases, that’s probably bad (because we’re trying to minimize loss). But if the loss decreases, that’s probably good.

<strong>In short, the goal of backprop is to measure how much each param affects the loss</strong>, which is encoded in the gradients. We then use the gradients in the “Step” phase to actually adjust those params and minimize our <em>estimated </em>loss<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>.

Note: This is where ‘gradient descent’ gets its name: we’re trying to move around in the params’ <em>gradient </em>space to <em>descend </em>to the lowest possible estimated loss.

<strong>Try to keep this goal in mind for the rest of the assignment</strong>: it should help contextualize things.

<strong>Autograd accomplishes this exact same thing, but it adds code to forward propagation in order to automate backprop. </strong>“Step” will be mostly unchanged, however, as it takes place after backprop/autograd anyway.

With this in mind, let’s walk through what autograd is doing during Forward Propagation (autograd’s <strong>“forward pass”</strong>) and Backpropagation (autograd’s <strong>“backward pass”</strong>).

<strong>0.1.4       Forward Pass</strong>

<table width="122">

 <tbody>

  <tr>

   <td width="24">com</td>

   <td width="15">pu</td>

   <td width="12">ta</td>

   <td width="36">tional</td>

   <td width="36"></td>

  </tr>

 </tbody>

</table>

During forward propagation, autograd automatically constructs a graph. It does this in the background, while the function is being run.

<strong>The computational graph tracks how elementary operations modify data throughout the function</strong>. Starting from the left, you can see how the input variables <strong>a </strong>and <strong>b </strong>are first multiplied together, resulting in a temporary output <strong>Op:Mult</strong>. Then, <strong>a </strong>is added to this temporary output, creating <strong>d </strong>(the middle node).

Nodes are added to the graph whenever an operation occurs. In practice, we do this by calling the .apply() method of the operation (which is a subclass of autograd engine.Function). Calling .apply() on the subclass implicitly calls Function.apply(), which does the following:

<ol>

 <li>Create a node object for the operation’s output</li>

 <li>Run the operation on the input(s) to get an output tensor</li>

 <li>Store information on the node, which links it to the comp graph</li>

 <li>Store the node on the output tensor</li>

 <li>Return the output tensor</li>

</ol>

It sounds complicated, but remember that all of this is happening in the background; all the user sees is that an operation occurred on some tensors and they got the correct output tensor.

<strong>But what information is stored on the node, and how does it link the node to the comp graph?</strong>

The node stores two things: <strong>the operation that created the node’s data</strong><a href="#_ftn3" name="_ftnref3"><sup>[3]</sup></a> and <strong>a record of the node’s “parents” </strong>(if it has any… ).

Recall that each tensor stores its own node. So when making the record of the current node’s parents, we can usually just grab the nodes from the input tensors. But also recall that we <em>only create nodes when an operation is called</em>. Op:Mult was the very first operation, so its input tensors a and b didn’t even have nodes initialized for them yet.

To solve this issue, Function.apply() also checks if any parents need to have their nodes created for them. If so, it creates the <em>appropriate type of node </em>for that parent (we’ll introduce node types during backward), and then adds that node to its list of parents. Effectively, this connects the current node to its parent nodes in the graph.

To recap, whenever we have an operation on a tensor in the comp graph, we create a node, get the output of the operation, link the node to the graph, and store the node on the output tensor.

Here’s the same graph again, for reference:

Below is the code that created this graph:

a = torch.tensor(1., requires_grad=True) b = torch.tensor(2.) c = torch.tensor(3., requires_grad=True)

d = a * b + a e = (d + c) + 3

And here’s that code translated into symbolic math, just to make clear the inputs and outputs:

<table width="404">

 <tbody>

  <tr>

   <td colspan="15" width="220"></td>

   <td colspan="22" width="387"><em>f</em>(<em>x,y,z</em>) = ((<em>x </em>· <em>y </em>+ <em>x</em>) + <em>z</em>) + 3</td>

   <td width="17">(1)</td>

  </tr>

  <tr>

   <td colspan="15" width="220"></td>

   <td colspan="22" width="387">Let <em>e </em>= <em>f</em>(<em>a,b,c</em>)</td>

   <td width="17">(2)</td>

  </tr>

  <tr>

   <td width="22">We</td>

   <td colspan="3" width="52">strongly</td>

   <td colspan="2" width="20">rec</td>

   <td width="18">om</td>

   <td colspan="2" width="35">mend</td>

   <td colspan="2" width="28">that</td>

   <td colspan="2" width="29">you</td>

   <td colspan="3" width="39">pause</td>

   <td width="24">and</td>

   <td width="24">try</td>

   <td width="20">to</td>

   <td colspan="2" width="32">recre</td>

   <td width="19">ate</td>

   <td colspan="2" width="24">the</td>

   <td colspan="2" width="39">above</td>

   <td colspan="2" width="40">graph</td>

   <td width="19">on</td>

   <td colspan="2" width="27">pen</td>

   <td colspan="2" width="27">and</td>

   <td width="20">pa</td>

   <td width="22">per,</td>

   <td colspan="2" width="28">just</td>

   <td width="17">by</td>

  </tr>

  <tr>

   <td width="22">look</td>

   <td width="23">ing</td>

   <td width="18">at</td>

   <td colspan="2" width="25">the</td>

   <td colspan="3" width="32">code</td>

   <td width="26">and</td>

   <td width="4"></td>

   <td width="24">the</td>

   <td width="14">de</td>

   <td colspan="2" width="27">scrip</td>

   <td colspan="2" width="26">tion</td>

   <td width="24">on</td>

   <td width="24">the</td>

   <td width="20">pre</td>

   <td width="10">vi</td>

   <td width="22">ous</td>

   <td colspan="2" width="36">page.</td>

   <td colspan="2" width="39">Track</td>

   <td colspan="2" width="35">what</td>

   <td width="11">in</td>

   <td width="19">for</td>

   <td width="17">ma</td>

   <td colspan="2" width="26">tion</td>

   <td colspan="2" width="31">each</td>

   <td width="22">ten</td>

   <td width="20">sor</td>

   <td colspan="2" width="25">and</td>

  </tr>

  <tr>

   <td width="22"></td>

   <td width="26"></td>

   <td width="23"></td>

   <td width="10"></td>

   <td width="18"></td>

   <td width="9"></td>

   <td width="20"></td>

   <td width="7"></td>

   <td width="31"></td>

   <td width="7"></td>

   <td width="25"></td>

   <td width="18"></td>

   <td width="14"></td>

   <td width="14"></td>

   <td width="4"></td>

   <td width="24"></td>

   <td width="29"></td>

   <td width="29"></td>

   <td width="23"></td>

   <td width="11"></td>

   <td width="25"></td>

   <td width="19"></td>

   <td width="20"></td>

   <td width="8"></td>

   <td width="35"></td>

   <td width="8"></td>

   <td width="30"></td>

   <td width="15"></td>

   <td width="23"></td>

   <td width="23"></td>

   <td width="8"></td>

   <td width="18"></td>

   <td width="13"></td>

   <td width="22"></td>

   <td width="25"></td>

   <td width="25"></td>

   <td width="8"></td>

   <td width="20"></td>

  </tr>

 </tbody>

</table>

each node is storing. Ignore node types for now. It’s ok to do this slowly, while asking clarifying questions on Piazza. This is confusing until you get it, and starting the code too early risks creating a debugging safari.

<strong>But why are we making this graph?</strong>

Remember: we’re doing all of this to calculate gradients. Specifically, <strong>the partial gradients of the loss w.r.t. each gradient-enabled tensor (any tensor with </strong>requires grad==True<strong>, see above code)</strong>. For us, our gradient-enabled tensors are a and c.

Goal:  and

<strong>Think of the graph as a trail of breadcrumbs that keeps track of where we’ve been. It’s used to retrace our steps back through the graph during backprop.</strong>

By the way, this is where ”backpropagation” gets its name: both autograd/backprop traverse graphs <em>backwards </em>while calculating gradients.

Why backwards? Find out on the next page…

traversing the entire graph in linear time<sup>4</sup>.

<strong>Doing this in reverse is just much more efficient than doing it forwards. Pause and try to imagine why this is the case. </strong>For reference, the answer is <a href="https://en.wikipedia.org/wiki/Automatic_differentiation#/Reverse_accumulation">here</a>

At each recursive call, Autograd calculates one gradient for <strong>each </strong>input.

<em>∂ </em>final              <em>∂ </em>final <em>∂ </em>output

=

<em>∂ </em>input<em><sub>i               </sub>∂ </em>output <em>∂ </em>input<em><sub>i</sub></em>

Each gradient is then passed onto its respective parent, but only if that parent is “gradient-enabled” (requires grad==True). If the parent isn’t, the parent does not get its own gradient/recursive call. Note: constants like the 3 node are <em>not </em>gradient-enabled by default.

Eventually, there will be a gradient-enabled node that has no more gradient-enabled parents. For nodes like these, all we do is store the gradient they just received in their tensor’s .grad.

The rules may sound complicated, but the results are easy to understand visually:

<strong>Let’s walk through the backward pass step-by-step</strong>. Calculations will be simpler than usual here, as they’re on scalars instead of matrices, but the main ideas are the same.

<strong>Step 1: Starting Traversal</strong>

We begin this process by calling e.backward(). This method simply calculates the gradient of e w.r.t. itself:

Remember that the gradient of a tensor w.r.t. itself is just a tensor of the same shape, filled with 1’s.

It then passes (without storing) its gradient to the graph traversal method: autograd engine.backward().

<strong>Step 2: First Recursive Call</strong>

We’re now in autograd engine.backward(), which performs the actual recursive DFS.

At each recursive call, we’re given two objects.

The first object is a tensor containing the gradient of the final node w.r.t. our output (here, <em><u><sup>∂</sup></u><sub>∂</sub></em><u><sup>e</sup></u><sub>e</sub>). We’ll use this to calculate our own gradient(s).

The second object “grad fn” contains the current node object. The node object has a method .apply() that contains instructions on what to do with the given gradient.

In this case, grad fn is a BackwardFunction object. So calling grad fn.apply(grad of outputs) actually calls the operation’s gradient calculation method (Add.backward()). This calculates the gradients w.r.t. each input:

<em>∂e              ∂e        ∂e</em>

=

<em>∂</em>Op:Add         <em>∂e ∂</em>Op:Add

Notice how the given gradient (here, ) was used in both calculations.

We now pass the gradients onto their respective parents for their own recursive calls, but only if they have requires grad==True. This means we only pass Op:Add its gradient, as [3] is a constant. No more work needs to be done for [3], as it has no parents and no need to store a gradient.

object again.

=

<em>∂c        ∂ </em>Op:Add          <em>∂c</em>

Same as before, we pass these gradients to our parents (c and d). But let’s explain <strong>c</strong>’s recursive call first, as it’s a bit different<a href="#_ftn4" name="_ftnref4"><sup>[4]</sup></a>.

<strong>Step 4: Storing a gradient</strong>

This call is a bit different, as this node’s tensor c is gradient-enabled (requires grad==True) and has no parents (is leaf==True).

As discussed before, this means that we have to store the gradient we just received. In NNs, tensors like c are usually network params.

We can actually store gradients by calling the same grad fn.apply() function, because this time grad fn is an AccumulateGrad object. AccumulateGrad.apply() handles storing gradients.

&gt;&gt; grad_fn.apply(grad_of_outputs)

&gt;&gt; c.grad tensor(1.)

Again, we’ve re-used the command grad fn.apply(), but its functionality changed depending on what grad fn was.

<strong>Step 5</strong>

Back to d’s call, you know the drill.

<em>∂ </em>Op:Mult

<strong>Step 6</strong>

We’ve stepped into the a node to store its gradient.

&gt;&gt; grad_fn.apply(gradient_e_wrt_a)

&gt;&gt; a.grad tensor(1.)

Now to step out and step into the Op:Mult node.

<strong>Step 7: Plot Twist!</strong>

Op:Mult is a bit different – thus far, our derivatives have only been over addition nodes. Now we have a new operation (multiplication) that has a different kind of derivative.

In general, for any <em>z </em>= <em>xy</em>, we know that  and . Here’s the difference: notice that <strong>the derivative of a product depends on the specific values of its inputs</strong>. For , we need to know <em>y</em>, and for , we need to know <em>x</em>. Similarly, for Op:Mult’s gradients, we’ll need to know a and b.

The problem is that we saw the inputs during forward, but now we’re in backward. To fix this, we store any information we need in an object that gets passed to both the forward and backward methods. In short, <strong>whenever we need to pass information between forward and backward, store it in the </strong>ContextManager <strong>object (“</strong>ctx<strong>”) during forward, and retrieve the info during backward.</strong>

<em>∂e          ∂ </em>Op:Mul

Op:Mul         <em>∂a</em>

Remember, b has requires grad==False, so we don’t pass its gradient. But for a…

<strong>Step 8: Plot Twist (again)!</strong>

We need to store this gradient, but remember that we already stored a.grad= in Step 6. What do we do with this new one?

Answer: We <strong>“accumulate” </strong>gradients with a += operation. This is handled by AccumulateGrad.apply().

&gt;&gt;&gt; a.grad tensor(1.)

&gt;&gt;&gt; grad_fn.apply(gradient_e_wrt_a)

&gt;&gt;&gt; a.grad tensor(3.)

Now <strong>a</strong>’s stored gradient is = 3.

We’re done! But before we finish: why +=? Why would we add these gradients? Find out on the next page…

<strong>0.1.6         Accumulating Gradients</strong>

Q: But why do we “accumulate” gradients with +=?

A: Because of the <strong>generalized chain rule</strong>.

<strong>Generalized Chain Rule</strong>

For a function <em>f </em>with scalar output <em>y</em>, where each of its <em>M </em>arguments are functions (<em>g</em><sub>1</sub><em>,g</em><sub>2</sub><em>,…,g<sub>M</sub></em>) of <em>N </em>variables (<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,…,x<sub>N</sub></em>):

<em>y </em>= <em>f</em>(<em>g</em><sub>1</sub>(<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,…,x<sub>N</sub></em>)<em>,g</em><sub>2</sub>(<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,…,x<sub>N</sub></em>)<em>…,g<sub>M</sub></em>(<em>x</em><sub>1</sub><em>,x</em><sub>2</sub><em>,…,x<sub>N</sub></em>))

the partial derivative of its output w.r.t. <strong>one input variable </strong><em>x<sub>i </sub></em>is:

Notice, we’re adding products of partial derivatives. You can interpret this as <strong>summing </strong><em>x<sub>i</sub></em><strong>’s various “pathways” (chain rule expansions) to influencing the output </strong><em>y</em>.

To illustrate, let’s say we want :

Notice that each summand is a single reverse-order “pathway” that leads from <em>x</em><sub>1 </sub>all the way to <em>y</em>. In autograd, every time a path finishes tracing back to the input, the term’s product is stored with a += until the other paths make it back. When all paths make it back to the node, all the terms have been summed, and its partial gradient is complete.

Autograd essentially does this, except it’s not just w.r.t. one variable, it’s <strong>the partial w.r.t. ALL gradient-enabled tensors at the same time</strong>. It does this because pathways often overlap; if we calculated each partial separately, we’d be re-calculating the same term multiple times, wasting computation.

Autograd’s pretty good!

<strong>0.1.7         Conclusion: Code</strong>

We’re done! Here were our results:

Let’s verify that this is correct using the real Torch.

<strong>Forward:</strong>

a = torch.tensor(1., requires_grad=True) b = torch.tensor(2.) c = torch.tensor(3., requires_grad=True)

d = a + a * b e = (d + c) + 3

<strong>Backward:</strong>

e.backward()

&gt;&gt;&gt; print(a.grad) tensor(3.)

&gt;&gt;&gt; print(b.grad) # No result returned, empty gradient

&gt;&gt;&gt; print(c.grad) tensor(1.)

Nice.

<strong>0.1.8          Conclusion: Symbolic Math</strong>

Here’s the equivalent symbolic math for each of the two gradients we stored.

Notice how much overlap there is between paths; autograd only computed that overlap once.

Again, autograd’s pretty good!

<h3>0.2       Autograd File Structure</h3>

It may be hard to keep track of all the autograd-specific code, so we’ve provided a quick guide below.

This file contains the forward/backward behavior of operations in the computational graph. In other words,

<table width="552">

 <tbody>

  <tr>

   <td width="24">any</td>

   <td width="17">op</td>

   <td width="11">er</td>

   <td width="6">a</td>

   <td width="26">tion</td>

   <td width="30">that</td>

   <td width="13">af</td>

   <td width="29">fects</td>

   <td width="24">the</td>

   <td width="21">net</td>

   <td width="43">work’s</td>

   <td width="27">final</td>

   <td width="26">loss</td>

   <td width="35">value</td>

   <td width="43">should</td>

   <td width="32">have</td>

   <td width="20">an</td>

   <td width="18">im</td>

   <td width="17">ple</td>

   <td width="24">men</td>

   <td width="12">ta</td>

   <td width="26">tion</td>

   <td width="28">here</td>

  </tr>

 </tbody>

</table>

<ol start="6">

 <li><sup>6</sup>. This also includes loss functions.</li>

</ol>

functional.py class Add…………………………………..[Given] Element-wise addition between tensors class Sub…………………………………………………..Same as above; subtraction class Mul………………………………………………….Element-wise tensor product class Div………………………………………………….Element-wise tensor division class Transpose……………………………………………[Given] Transposing a tensor class Reshape……………………………………………….[Given] Reshaping a tensor class Log……………………………………………[Given] Element-wise log of a tensor function cross entropy()………………….Cross-entropy loss (loss function in comp graph)

Note: elementary operators are generally subclasses of autograd engine.Function<sup>7</sup>.

<strong>0.2.2 </strong>autograd engine.py

autograd engine.py function backward()………………………………………………..The recursive DFS class Function………………………………….. Base class for functions in functional.py class AccumulateGrad………..Used in backward(). Represents nodes that accumulate gradients class ContextManager…………Arg ”ctx” in functions. Passes data between forward/backward class BackwardFunction…………………Used in backward(). Represents intermediate nodes

In this file you’ll only need to implement backward() and Function.apply(). But you’ll want to read the other classes’ code, as you’ll be extensively working with them.

<strong>0.2.3 </strong>tensor.py

tensor.py class Tensor………………………..Wrapper for np.array, allows interaction with MyTorch

Contains only the class representing a Tensor. Most operations on Tensors will likely need to be defined as a class method here<sup>8</sup>.

Pay close attention to the class variables defined in init (). Appendix A covers these in detail; especially before starting problem 1.2, we highly encourage you to read it.

<sup>6</sup>See <a href="https://pytorch.org/docs/stable/nn.functional.html">the actual Torch’s</a> <a href="https://pytorch.org/docs/stable/nn.functional.html">nn.functional</a>              for ideas on what operations belong here

<sup>7</sup>In Python, subclasses indicate their parent in their declaration. i.e. class Add(Function) <sup>8</sup>Again, see <a href="https://pytorch.org/docs/stable/tensors.html##torch.Tensor">the actual</a> <a href="https://pytorch.org/docs/stable/tensors.html##torch.Tensor">torch.Tensor</a>        for ideas

<h3>0.3       Running/Submitting Code</h3>

This section covers how to test code locally and how to create the final submission.

<strong>Note that there are two different local autograders. </strong>We’ll explain them both here.

<strong>0.3.1            Running Local Autograder (Before MNIST)</strong>

Run the command below to calculate scores for Problems 1.1 to 2.6 (all problems before Problem 3: MNIST)

./grade.sh 1

If this doesn’t work, converting <a href="https://en.wikipedia.org/wiki/Newline">line-endings</a>         may help:

sudo apt install dos2unix dos2unix grade.sh

./grade.sh 1

If all else fails, you can run the autograder manually with this:

python3 ./autograder/hw1_autograder/runner.py

<strong>Note: as MNIST is not autograded, 100/110 is a ”full” score for this section.</strong>

<strong>0.3.2            Running Local Autograder (MNIST only)</strong>

After completing all of Problem 3, use this autograder to test MNIST and generate plots.

./grade.sh m

You can also run it manually with this:

python3 ./autograder/hw1_autograder/test_mnist.py

<strong>Note</strong>: If you’re using WSL, plotting may not work unless you run VcXsrv or Xming; see <a href="https://stackoverflow.com/questions/43397162/show-matplotlib-plots-and-other-gui-in-ubuntu-wsl1-wsl2">here</a>

<strong>0.3.3         Running the Sandbox</strong>

We’ve provided sandbox.py: a script to test and easily debug basic operations and autograd. When you add your own new operators, write your own tests for these operations in the sandbox.

python3 sandbox.py

<strong>0.3.4         Submitting to Autolab</strong>

<strong>Note: You can submit to Autolab even if you’re not finished yet. You should do this early and often, as it guarantees you a minimum grade and helps avoid last-minute problems with Autolab.</strong>

Run this script to gather the needed files into a handin.tar file:

./create_tarball.sh

You can now upload handin.tar to <a href="https://autolab.andrew.cmu.edu/courses/11485-f20/assessments">Autolab</a>              <a href="https://autolab.andrew.cmu.edu/courses/11485-f20/assessments">.</a>

<table width="625">

 <tbody>

  <tr>

   <td width="17">To</td>

   <td width="13">re</td>

   <td width="30">ceive</td>

   <td width="37">credit</td>

   <td width="20">for</td>

   <td width="28">prob</td>

   <td width="23">lem</td>

   <td width="14">3,</td>

   <td width="24">you</td>

   <td width="32">must</td>

   <td width="12">al</td>

   <td width="34">ready</td>

   <td width="30">have</td>

   <td width="32">plots</td>

   <td width="22">gen</td>

   <td width="11">er</td>

   <td width="27">ated</td>

   <td width="18">by</td>

   <td width="22">the</td>

   <td width="48">MNIST</td>

   <td width="16">au</td>

   <td width="12">to</td>

   <td width="43">grader.</td>

   <td width="35">Make</td>

   <td width="26">sure</td>

  </tr>

 </tbody>

</table>

the plot image is in the hw1 folder. The plots will be included automatically the next time you generate a submission.

<h1>1         Implementing Autograd [Total: 40 points]</h1>

We’ll start implementing autograd by programming some elementary operations.

<strong>Advice</strong>:

<ol>

 <li>We’ve provided py (optional) to help you easily debug operations and the rest of autograd.</li>

 <li>Don’t change the names of existing classes/variables. This will confuse the Autograder.</li>

 <li>Use existing <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html">NumPy functions</a> <a href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html">,</a> especially if they replace loops with vector operations.</li>

 <li>Debuggers like <a href="https://docs.python.org/3/library/pdb.html">pdb</a> are usually simpler and more reliable than print() statements<a href="#_ftn5" name="_ftnref5"><sup>[5]</sup></a>. Also, see this recitation on debugging: <a href="http://deeplearning.cs.cmu.edu/F20/index.html#recitations">Recitation 0F</a></li>

 <li>Read error messages closely, and feel free to read the autograder’s code for context/clues.</li>

</ol>

<h2>1.1      Basic Operations [15 points]</h2>

Let’s begin by defining how some elementary tensor operations behave in the computational graph.

<strong>1.1.1         Element-wise Subtraction [5 points]</strong>

In nn/functional.py, we’ve given you the Add class as an example. Now, complete Sub.

Then, in tensor.py, complete Tensor. sub (). This function defines what happens when you try to subtract two tensors like this: tensor a – tensor b<sup>10</sup>. By calling F.Sub here, we’re connecting every “-” operation to the computational graph.

<strong>1.1.2         Element-wise Multiplication [5 points]</strong>

In nn/functional.py, complete Mul. Then create/complete Tensor. mul ()

<strong>1.1.3         Element-wise Division [5 points]</strong>

In nn/functional.py, complete Div. Then create/complete Tensor. truediv ().

<h2>1.2      Autograd [25 points]</h2>

Now to implement the core Autograd engine. If you haven’t yet, we encourage you to read Appendix A. Note that while implementing this, you may temporarily break the basic operations tests, so plan to debug.

<ul>

 <li>In autograd engine.py finish apply(). During the forward pass, this both runs the called operation AND adds node(s) onto the computational graph. This method is called whenever an operation occurs (usually in tensor.py). We’ve given you some starting code; see the intro section for hints on completing this.</li>

 <li>In py, implement the Tensor.backward() function. This should be very short – it kicks off the DFS. See step 1 of the backward example for what this is doing.</li>

 <li>In autograd engine.py, implement the backward(grad fn, grad of output) This is the DFS backward pass. Think about what objects are being passed, and the base case(s) of the recursion. This code should also be short.</li>

</ul>

<h1>2        MLPs [Total: 60 points]</h1>

Now for the fun stuff – coding a trainable MLP.

This will involve coding network layers, an activation function, and an optimizer. Only after completing these tasks three will you be able to challenge the formidable MNIST .

Note: from now on, you will need to implement your OWN operations/functions if you need them. We have given you some freebies, however.

A common question is whether an operation should account for multiple dimensions or other edge cases. A good rule of thumb is to implement what you need, and then modify the code later when needed. Also, avoid implementing any operations you don’t need. Be sure to keep using the sandbox to develop/debug.

<h2>2.1      Linear Layer [10 points]</h2>

First, in nn/sequential.py, implement Sequential.forward(). This class simply passes input data through each layer held in Sequential.layers and returns the final result<a href="#_ftn6" name="_ftnref6"><sup>[6]</sup></a>.

Next, in nn/linear.py, complete Linear.forward(). <strong>This will require implementing matrix operation(s) in </strong>nn/functional.py. Hint: what operations are in the formula below?

Linear(x) = <em>xW<sup>T </sup></em>+ <em>b</em>

(Note: this formulation should be cleaner than the commonly used <em>Wx </em>+ <em>b</em>.)

Note that you will have to modify Add to support tensors with different shapes. This is done with <a href="https://numpy.org/doc/1.18/user/basics.broadcasting.html">broadcasting</a> <a href="https://numpy.org/doc/1.18/user/basics.broadcasting.html">. </a>While NumPy will handle broadcasting for you in forward, you’ll need to implement the derivative of the broadcast during the backward.

For example, say we add a tensor A, shaped (3, 4), with B, shaped (4,). NumPy will turn B into (3, 4) implicitly in forward. However in backward, the gradient we calculate with respect to B will be shaped (3, 4); it needs to be reduced to shape (4,) before we return it.

To do this, we recommend you implement a function unbroadcast(grad, shape) in functional.py. In theory, this function should be able to unbroadcast gradients for ANY operation (not just Add, but also Sub, Mult, Etc). However, if you’d prefer to implement just a limited 2D unbroadcast for Add, that will work for now.

In sandbox.py, you can use testbroadcast to test your solution to unbroadcasting.

<h2>2.2      ReLU [5 points]</h2>

First, in nn/functional.py, create and complete a ReLU(Function) class. Similar to the elementary operations before, this class describes how the ReLU activation function works in the computational graph.

( <em>z   z &gt; </em>0

ReLU(z) =

0     <em>z </em>≤ 0

ReLU

Then, in nn/activations.py, complete ReLU.forward() by calling the functional.ReLU function, just like you did with operations in tensor.py.

<h2>2.3         Stochastic Gradient Descent (SGD) [10 points]</h2>

In optim/sgd.py, complete the SGD.step() function.

After gradients are calculated, optimizers like SGD are used to update trainable params in order to minimize loss.

Note that this class inherits from optim.optimizer.Optimizer. Also, make sure this code does NOT add to the computational graph.

<em>W</em><em>k </em>= <em>W</em><em>k</em>−1 − <em>η</em>∇<em>WLoss</em>(<em>W</em><em>k</em>−1)

<h2>2.4         Batch Normalization (BatchNorm) [BONUS]</h2>

Note that this problem is being converted to a bonus. You may skip it for now; see Piazza for details.

In nn/batchnorm.py, complete the BatchNorm1d class.

<strong>For a conceptual introduction to BatchNorm, see Appendix C.</strong>

BatchNorm (Ioffe and Szegedy 2015) uses the following equations for its forward function:

(3)

(4)

(5)

(6)

Note that you will want to convert these equations to matrix operations.

<em>x<sub>i </sub></em>is the input to the BatchNorm layer. Within the forward method, compute the sample mean (<em>µ</em><sub>B</sub>), sample variance (<em>s</em><sup>2</sup><sub>B</sub>), and norm (<strong>x</strong>ˆ<em><sub>i</sub></em>). Epsilon () is used when calculating the norm to avoid dividing by zero. Lastly, return the final output (<em>y<sub>i</sub></em>).

You may need to implement operation(s) like Sum in nn/functional.py and tensor.py. Remember to use matrix operations instead of for loops; loops are too slow.

Also, you’ll need to calculate the <strong>running mean/variance too</strong>:

(7)

(8)

(9)

Note: The notation above is consistent with PyTorch’s implementation of Batchnorm. The <em>α </em>above is actually 1 − <em>α </em>in the original paper.

BatchNorm operates differently during training and eval. During training (BatchNorm1d.is train==True), your forward method should calculate an unbiased estimate of the variance (<em>σ</em><sub>B</sub><sup>2 </sup>), and maintain a running average of the mean and variance. These running averages should be used during inference (BatchNorm1d.is train==False) in place of <em>µ</em><sub>B </sub>and <em>s</em><sup>2</sup><sub>B</sub>.

<h2>2.5       Cross Entropy Loss [15 points]</h2>

<strong>For info on how to calculate Cross Entropy Loss, see Appendix C</strong>.

There are two ways that you can complete this question and receive full credit.

<ol>

 <li>You can complete the cross entropy function we provided in py. In this case, autograd will handle the backward calculation for you.</li>

 <li>You can create and complete your own subclass of Function, just like the operations earlier. This means you’ll need to implement the derivative calculation, which we describe in the appendix. You’ll also need to change loss.CrossEntropyLoss.forward() to call this subclass.</li>

</ol>

As long as your implementation is correct AND is correctly called by nn.loss.CrossEntropyLoss.forward(), you will receive full credit.

<h2>2.6      Momentum [10 points]</h2>

In optim/sgd.py modify SGD.step() to include ”momentum”. For a good explanation of momentum, see <a href="https://ruder.io/optimizing-gradient-descent/index.html#momentum">here</a>        <a href="https://ruder.io/optimizing-gradient-descent/index.html#momentum">.</a>

We will be using the following momentum update equation:

∇<em>W<sup>k </sup></em>= <em>β</em>∇<em>W<sup>k</sup></em><sup>−1 </sup>− <em>η</em>∇<em><sub>W</sub>Loss</em>(<em>W<sup>k</sup></em><sup>−1</sup>)

<em>W</em><em>k </em>= <em>W</em><em>k</em>−1 + ∇<em>W</em><em>k</em>

Note that you’ll have to use self.momentums, which tracks the momentum of each parameter. Make sure this code does NOT add to the computational graph.

<h1>3        MNIST</h1>

Finally, after all this, it’s time to print(“Hello world!”). But first, some concepts.

<strong>MNIST</strong>. Each observation in MNIST is a (28×28) grayscale image of a handwritten digit [0-9] that’s been flattened into a 1-D array of floats between [0,1]. Your task is to identify which digit is in each image. You’re also given the true label (int in [0,9]) for each observation.

<strong>Batching</strong>. In DL, instead of training on one observation at a time, we usually train on small, evenly-sized groups of points that we call “batches”. Ideally (and generally in practice), this stabilizes training by decreasing the variation between individual data points.

During forward(), we put a single batch into a tensor and pass it through the model. This means we end up with a vector of losses: one loss value for each training point. We then aggregate this vector to create a single loss value (usually by averaging or summing). <em>Your implementation of XELoss already handles aggregation by averaging; just use this as is</em>.

<strong>Train/Validation/Test</strong>. <a href="https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets">Review this to understand your upcoming task</a> <a href="https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets">.</a> Today, we’re only implementing training and validation routines. The autograder already has pre-split train/val data, and will also handle the plotting.

<h2>3.1      Initialize Objects</h2>

In hw1/mnist.py, complete mnist(). Initialize your criterion (CrossEntropyLoss), your optimizer (SGD), and your model (Sequential).

Set the learning rate of your optimizer to lr=0.1.

Use the following architecture:

Linear(784, 20) -&gt; BatchNorm1d(20) -&gt; ReLU() -&gt; Linear(20, 10)

Finally, call the train() method, which we’ll implement below.

<h3>3.2 train()</h3>

Next, implement train(). Some notes:

<ol>

 <li>We’ve preset batch size=100 and num epochs=3. Feel free to adjust while developing/testing.</li>

 <li>Make sure to shuffle the data at the start of each epoch (hint: random.shuffle())</li>

 <li>Make sure that input/label tensors are NOT gradient enabled.</li>

 <li>For the sake of visualization, perform validation after every 100 batches and store the accuracy. Normally, people validate once per epoch, but we wanted to show a detailed curve.</li>

</ol>

Pseudocode:

def train():

model.activate_train_mode() for each epoch:

shuffle_train_data() batches = split_data_into_batches() for i, (batch_data, batch_labels) in enumerate(batches): optimizer.zero_grad() # clear any previous gradients out = forward_pass(batch_data) loss = criterion(out, batch_labels) loss.backward() optimizer.step() # update weights with new gradients if i is divisible by 100: accuracy = validate() store_validation_accuracy(accuracy) model.activate_train_mode()

return validation_accuracies

(this is a typical routine; will become familiar throughout the semester)

<h3>3.3 validate()</h3>

Finally, implement validate(). Pseudocode again:

def validate():

model.activate_eval_mode() batches = split_data_into_batches() for (batch_data, batch_labels) in batches:

out = forward_pass(batch_data)

batch_preds = get_idxs_of_largest_values_per_batch(out) num_correct += compare(batch_preds, batch_labels)

accuracy = num_correct / len(val_data) return accuracy

<h2>Plotting and Submission</h2>

After completing the above methods, you can run the MNIST autograder (NOT the regular autograder; see Section 0.3) to test your script and generate a plot that shows the val accuracy at each epoch. The plot will be stored as ‘validation accuracy.png’.

Note: Make sure this image is in the hw1 folder; it needs to be here to be included in the submission. Older

<table width="379">

 <tbody>

  <tr>

   <td width="18">ver</td>

   <td width="30">sions</td>

   <td width="15">of</td>

   <td width="23">the</td>

   <td width="31">hand</td>

   <td width="21">out</td>

   <td width="29">may</td>

   <td width="24">not</td>

   <td width="16">au</td>

   <td width="12">to</td>

   <td width="25">mati</td>

   <td width="31">cally</td>

   <td width="34">place</td>

   <td width="34">them</td>

   <td width="36">there.</td>

  </tr>

 </tbody>

</table>

The plot should look something like this:

<strong>Note: the plot must be in your final submission to receive credit for Problem 3. </strong>When you run the autograder, the image should automatically be placed in the main folder. Then, running the submission generator will automatically place the image in submission. We’ll grade it manually afterwards.

<strong>You’re done! </strong>Implementing this is seriously no easy task. Very few people have implemented automatic differentiation from scratch. Congrats and great work. More not-easy tasks to come .

<h1>References</h1>

Ioffe, Sergey and Christian Szegedy (2015). “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”. In: <em>CoRR </em>abs/1502.03167. arXiv: <a href="https://arxiv.org/abs/1502.03167">1502.03167.</a> url: <a href="https://arxiv.org/abs/1502.03167">http://arxiv. </a><a href="https://arxiv.org/abs/1502.03167">org/abs/1502.03167.</a>

<strong>Appendix</strong>

<h1>A        Tensor Attributes</h1>

These are the class variables defined in Tensor. init (): class Tensor data (np.array) requires grad (boolean) is leaf (boolean) grad fn (some node object or None) grad (Tensor) is parameter (boolean)

<h2>A.1 requires grad and is leaf</h2>

The combination of these two variables determine the tensor’s node type. Specifically, during Function.apply(), you’ll be using these params to determine what node to create for the parent.

<strong>is leaf </strong>(default: True) indicates whether this tensor is a “<strong>Leaf Tensor</strong>”. Leaf Tensors are defined as not having any gradient-enabled parents. In short, any node that has requires grad=False is a Leaf Tensor<a href="#_ftn7" name="_ftnref7"><sup>[7]</sup></a>. <strong>requires </strong><strong>grad</strong><sup>13 </sup>(default: False) indicates whether gradients need to be calculated for this tensor.

The combination of these variables determines the tensor’s role in the computational graph:

<ol>

 <li><strong>AccumulateGrad </strong> Has is leaf=True and requires grad=True. This node is a gradientenabled node that has no parents. The .apply() of this node handles accumulating gradients in the tensor’s .grad.</li>

 <li><strong>BackwardFunction </strong> Has is leaf=False and requires grad=True. This is an intermediate node, where gradients are calculated and passed, but not stored. The .apply() of this node calculates the gradient(s) w.r.t. the inputs and returns them.</li>

 <li><strong>Constant </strong>node (Store a None in the parent list). Has is leaf=True and requires grad=False. This means this Tensor is a user-created tensor with no parents, but does not require gradient storage. This would be something like input data.</li>

</ol>

Remember, during Function.apply(), we’re creating and storing these nodes in (Tensor.grad fn).

Note: if any of a node’s parents requires grad, this node will also require grad.

This is so that autograd knows to pass gradients onto gradient-enabled parents. For example, see the diagram in Section 0.1.4. Op:Mult would have requires grad==True, because at least one of his parents (a) requires grad. But if all parents aren’t gradient enabled, a child would have requires grad=False and C.is leaf=True.

<h2>A.2 is parameter</h2>

Indicates whether this tensor contains parameters of a network. For example, Linear.weight is a tensor where is parameter should be True. NOTE: if is parameter is True, requires grad and is leaf must also be True.

<h1>B           Cross Entropy Loss (”XE Loss”)</h1>

For quick info, <a href="https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html">the official Torch doc is very good</a>           <a href="https://pytorch.org/docs/master/generated/torch.nn.CrossEntropyLoss.html">.</a> But we’ll go in depth here.

Let’s begin with a broad, coder-friendly definition of XE Loss.

If you’re trying to predict what class an input belongs to, XE Loss is a great loss function<a href="#_ftn8" name="_ftnref8"><sup>[8]</sup></a>. For a single training example, XE Loss essentially measures the “incorrectness” (<strong>“divergence”</strong>) of your confidence in the true label. The higher your loss, the more incorrect your confidence was.

To use XE Loss, the output of your network should be a float tensor of size (<em>N,C</em>), where <em>N </em>is batch size and <em>C </em>is the number of possible classes. We call this output a tensor of <strong>“logits”</strong>. The logits represent your unnormalized confidence that an observation belongs to each label. The label with the “highest confidence” (largest value in row) is usually your “prediction” for that observation<a href="#_ftn9" name="_ftnref9"><sup>[9]</sup></a>.

We’ll also need another tensor containing the <strong>true labels </strong>for each observation in the batch. This is a long<sup>16 </sup>tensor of size (<em>N,</em>), where each entry contains an index of the correct label.

There are essentially two steps to calculate XE Loss, which we’ll cover below.

<strong>NOTE: Directly implementing the below formulas with loops will be too slow. Convert to matrix operations and/or use NumPy functions.</strong>

<h2>Step 1: LogSoftmax</h2>

First, it applies a <a href="https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html">LogSoftmax()</a>                   to the logits. For a <strong>single observation</strong>:

<em>x</em><em>n </em>LogSoftmax(

Remember, the above formula is for a single observation. You need to do this for all observations in the batch. Also, <strong>don’t directly implement the above, as it’s numerically unstable. Read section B.1 for how to implement it.)</strong>

Softmax scales the values in a 1D vector into “probabilities” that sum up to 1. We then scale the values by applying the log. The resulting vector <em>p<sub>n </sub></em>contains floats that are ≤ 0.

<h2>Step 2: NLLLoss</h2>

Second, it calculates the <a href="https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html">Negative Log-Likelihood Loss</a> (NLLLoss) using the tensor from the previous step (<em>P</em>) and the true label tensor (<em>L</em>).

NLLLoss(

For the numerator, you’re summing the values at the correct indices. The <em>N </em>in the denominator is the batch size. Essentially, for a batch of outputs, we’re getting our average confidence in the correct answer.

That’s it! Calling NLLLoss(LogSoftmax(logits), labels) will give you your final loss. Note that it’s averaged across the batch.

<h2>B.1          Stabilizing LogSoftmax with the LogSumExp Trick</h2>

When implementing LogSoftmax, you’ll need the <strong>LogSumExp </strong>trick. This technique is used to prevent numerical underflow and overflow which can occur when the exponent is very large or very small. For example:

As you can see, for exponents that are too large, Python throws an overflow error, and for exponents that are too small, it rounds down to zero.

We can avoid these errors by using the LogSumExp trick:

<em>N                                                N</em>

log X <em>e</em><em>x</em><em>n </em>= <em>a </em>+ log X <em>e</em><em>x</em><em>n</em>−<em>a</em>

<em>n</em>=0                                            <em>n</em>=0

You can read proofs of its equivalence <a href="https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/">here</a>         and <a href="https://blog.feedly.com/tricks-of-the-trade-logsumexp/">here</a>

<h2>B.2         XE Loss – Derivation and Derivative</h2>

This section contains conceptual background and the derivative of XE Loss (which you may find useful).

Cross-entropy comes from information theory, where it is defined as the expected information quantified as  of some subjective distribution <em>Q </em>over an objective distribution <em>P</em>.

Simply put, it tells us how far our model is from the true distribution <em>P</em>. It does this by measuring how much information we receive when receiving new observations from <em>Q</em>.

Cross-Entropy is fully minimized when <em>P </em>= <em>Q</em>. The value of cross-entropy in this case is known simply as the entropy.

<em>H </em>

This is the irreducible information we receive when we observe the outcome of a random process.

For example, consider a coin toss. Even if we know the probability of heads (Bernoulli parameter: <em>p</em>) ahead of time, we don’t know what the result will be until we observed it. The greater <em>p </em>is, the more certain we are about the outcome and the less information we expect to receive upon observation.

When calculating XELoss, we first use the softmax function to normalize our network’s outputs before calculating the loss. The softmax outputs represent <em>Q</em>, our subjective distribution. We will denote each softmax output as ˆ<em>y<sub>j </sub></em>and represent the true distribution <em>P </em>with output labels <em>y<sub>j </sub></em>= 1 when the label is for each output. We let <em>y<sub>j </sub></em>= 1 when the label is <em>j </em>and <em>y<sub>j </sub></em>= 0 otherwise.

Next, we use Cross-Entropy as our objective function. The result is a degenerate distribution that will aim to estimate <em>P </em>when averaged over the training set.

Note that when we take the partial derivative of CrossEntropyLoss, we get the following result:

<strong>NOTE: Implementing the above directly may not give you the correct result. Remember, you averaged over the batch during </strong>Softmax() <strong>by dividing by </strong><em>N</em>.

This derivative is pleasingly simple and elegant. Remember, this is the derivative of softmax with crossentropy divergence with respect to the input. What this is telling us is that when <em>y<sub>j </sub></em>= 1, the gradient is negative; thus the opposite direction of the gradient is positive.

In short, it is telling us to increase the probability mass of that specific output through the softmax.

<h1>C        BatchNorm</h1>

Batch Normalization (“BatchNorm”) is a wildly successful technique for improving the speed and quality of learning in NNs. It does this by attempting to address an issue called <strong>internal covariate shift</strong>.

We encourage you to read the <a href="https://arxiv.org/abs/1502.03167">original paper</a> <a href="https://arxiv.org/abs/1502.03167">;</a> it’s written very clearly, and walks you through the math and reasoning behind each decision.

<h2>C.1         Motivation: Internal Covariate Shift</h2>

Internal covariate shift happens while training an NN.

In an MLP, each layer is training based on the activations of previous layer(s). But remember – <strong>previous layers are ALSO training, changing their weights and outputs all the time</strong>. This means that the later layers are working with frequently shifting information, leading to less stable and significantly slower training. This is especially true at the beginning of training, when parameters are changing a lot.

That’s internal covariate shift. It’s like if your boss AND your boss’s boss joined the company on the same day that you did. And they also have the same level of experience that you do. Now you’ll never get into FAANG…

<h2>C.2        Intro to BatchNorm</h2>

BatchNorm essentially introduces normalization<a href="https://en.wikipedia.org/wiki/Whitening_transformation">/whitening</a> <em>between </em>layers to help mitigate this problem. Specifically, a BN layer aims to linearly transform the output of the previous layer s.t. across the entire dataset, each neuron’s output has mean=0 and variance=1 AND is linearly decorrelated with the other neurons’ outputs.

By ‘linearly decorrelated’, we mean that for a layer <em>l </em>with <em>m </em>units, individual unit activities <strong>x </strong>= {<strong>x</strong><sup>(<em>k</em>)</sup><em>,…,</em><strong>x</strong><sup>(<em>d</em>)</sup>} are independent of each other – {<strong>x</strong><sup>(1) </sup>⊥ <em>…</em><strong>x</strong><sup>(<em>k</em>) </sup><em>… </em>⊥ <strong>x</strong><sup>(<em>d</em>)</sup>}. Note that we consider the unit activities to be random variables.

<strong>In short, we want to make sure that normalization/whitening for a single neuron’s output is happening consistently across the entire dataset</strong>. In truth, this is not computationally feasible (you’d have to feed in the entire dataset at once), nor is it always fully differentiable. So instead, we maintain “running estimates” of the dataset’s mean/variance and update them as we see more observations.

How do we do this? Remember that we’re training on batches<a href="#_ftn10" name="_ftnref10"><sup>[10]</sup></a> – small groups of observations usually sized 16, 32, 64, etc. <strong>Since each batch contains a random subsample of the dataset, we assume that each batch is somewhat representative of the entire dataset</strong>. Based on this assumption, we can use their means and variances to update our running estimates.

<h2>C.3        Implementing BatchNorm</h2>

This section gives more detail/explanation about the implementation of BatchNorm. Read it if you need any clarifications.

Given this setup, consider <em>µ</em><sub>B </sub>to be the mean and <em>σ</em><sub>B</sub><sup>2 </sup>the variance of a unit’s activity over the batch B<a href="#_ftn11" name="_ftnref11"><sup>[11]</sup></a>. For a training set X with <em>n </em>examples, we partition it into <em>n/m </em>batches B of size <em>m</em>. For an arbitrary unit <em>k</em>, we compute the batch statistics <em>µ</em><sub>B </sub>and <em>σ</em><sub>B</sub><sup>2 </sup>and normalize as follows:

(10)

(11)

<strong>x</strong><em>i </em>− <em>µ</em>B

<strong>x</strong>ˆ ←                                                                                      (12)

Note that we add in order to avoid dividing by zero.

A significant issue posed by simply normalizing individual unit activity across batches is that it limits the set of possible network representations. In order to avoid this, we introduce a set of trainable parameters (<em>γ</em><sup>(<em>k</em>) </sup>and <em>β</em><sup>(<em>k</em>)</sup>) that learn to make the BatchNorm transformation into an identity transformation.

To do this, these per-unit learnable parameters <em>γ</em><sup>(<em>k</em>) </sup>and <em>β</em><sup>(<em>k</em>) </sup>rescale and reshift the normalized unit activity. Thus the output of the BatchNorm transformation for a data example, <strong>y</strong><em><sub>i </sub></em>is:

<strong>y</strong><em><sub>i </sub></em>← <em>γ</em><strong>x</strong>ˆ<em><sub>i </sub></em>+ <em>β</em>

<strong>Training Statistics</strong>

<table width="413">

 <tbody>

  <tr>

   <td width="390"><em>E</em>[<em>x</em>] = (1 − <em>α</em>) ∗ <em>E</em>[<em>x</em>] + <em>α </em>∗ <em>µ</em><sub>B</sub></td>

   <td width="24">(13)</td>

  </tr>

  <tr>

   <td width="390"><em>V ar</em>[<em>x</em>] = (1 − <em>α</em>) ∗ <em>V ar</em>[<em>x</em>] + <em>α </em>∗ <em>σ</em><sub>B</sub><sup>2</sup></td>

   <td width="24">(14)</td>

  </tr>

 </tbody>

</table>

Note: The notation above is consistent with PyTorch’s implementation of Batchnorm. The <em>α </em>above is actually 1 − <em>α </em>in the original paper.

This is the running mean <em>E</em>[<em>x</em>] and running variance <em>V ar</em>[<em>x</em>] we talked about. We need to calculate them during training time when we have access to training data, so that we can use them to estimate the true mean and variance across the entire dataset.

If you didn’t do this and recalculated running means/variances during test time, you’d end up wiping the data out (mean will be itself, var will be inf) because you’re typically only shown one example at a time during test. This is why we use the running mean and variance.

<a href="#_ftnref1" name="_ftn1">[1]</a> See recitation 2 for explanation of the difference between derivatives/gradients

<a href="#_ftnref2" name="_ftn2">[2]</a> Why <em>estimated </em>loss? Try to consider what a “true” loss for a real world problem might look like.

<a href="#_ftnref3" name="_ftn3">[3]</a> In the code, we’ve already stored this for you.

<a href="#_ftnref4" name="_ftn4">[4]</a> In practice, it won’t matter what order you call the parents in

<a href="#_ftnref5" name="_ftn5">[5]</a> For an easy breakpoint, this oneliner is great: import pdb; pdb.set trace() <sup>10</sup><a href="https://docs.python.org/3/library/operator.html">See here for more info</a>

<a href="#_ftnref6" name="_ftn6">[6]</a> Sequential can be used as a simple ”model” or to group layers into a single module as part of a larger model. Useful for simplifying code.

<a href="#_ftnref7" name="_ftn7">[7]</a> It’s impossible for a tensor to have both requires grad=False and is leaf=False, hence 3 possible node types. <sup>13</sup>Official description of requires grad <a href="https://pytorch.org/docs/stable/notes/autograd.html">here</a>

<a href="#_ftnref8" name="_ftn8">[8]</a> It’s quite popular and commonly used in many different applications.

<a href="#_ftnref9" name="_ftn9">[9]</a> During val/test, the index of the <strong>maximum value </strong>in the row is usually returned as your label. <sup>16</sup>The official Torch uses long; for us, it should be ok to use int

<a href="#_ftnref10" name="_ftn10">[10]</a> Technically, <em>mini-batches</em>. “Batch” actually refers to the entire dataset. But colloquially and even in many papers, “batch” means “mini-batch”.

<a href="#_ftnref11" name="_ftn11">[11]</a> Again, technically ‘mini-batch’