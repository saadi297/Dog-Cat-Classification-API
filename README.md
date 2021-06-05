# Dog-Cat-Classification-API
## Requirements
Install numpy, tensorflow and flask. For tensorflow installation, see https://www.tensorflow.org/install/pip#virtual-environment-install
<pre><code>$ pip install numpy</code>
<code>$ pip install --upgrade tensorflow</code>
<code>$ pip install flask</code></pre>
## Download and Unzip Training Data
<pre><code>curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip</code>
<code>unzip -q kagglecatsanddogs_3367a.zip</code></pre>
## Remove Corrupt Imges
<pre><code>$ python filter_images.py </code></pre>
## Training

<pre><code>$ python main.py</code></pre>

## Testing 
<pre><code>python main.py --mode "test" --checkpoint_path saved_models/save_at_29.h5</code></pre> 
This model acheived 96% accuracy.
## Run API
Open terminal and run
<pre><code>$ python api.py</code></pre>
Open another terminal and run
<pre><code>$ curl --request POST 'http://localhost:5000/predict' --data-raw '{"image_path": "PetImages/Cat/6779.jpg"}'</code></pre>