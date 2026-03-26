import argparse
import os
import shutil
import textwrap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Create a parser
parser = argparse.ArgumentParser(
    description='Create a summary image from generated images.')
parser.add_argument('-output_dir',
                    type=str,
                    required=True,
                    help='The directory containing the generated images.')

parser.add_argument('-model_steps',
                    type=int,
                    required=True,
                    help='The number of steps the model was trained for.')

# Parse command line parameters
args = parser.parse_args()

output_dir = args.output_dir
model_steps = args.model_steps

prompt_dict = {
    
    "cigar":
    "sdss, smooth, cigar shaped galaxy",
    "in_between":
    "sdss, smooth, in-between round galaxy",
    "compelete_round":
    "sdss, smooth, completely round galaxy",
    "spiral_2_arm":
    "sdss, spiral galaxy, obvious bulge prominence, tightly wound spiral arms, 2 spiral arms",
    "edge_on":
    "sdss, edge-on galaxy, with rounded edge-on bulge",
    "merger_in-between": 
    "sdss, smooth, in-between round galaxy, a merger",
    "merger_spiral": 
    "sdss, spiral galaxy, just noticeable bulge prominence, tightly wound spiral arms, 2 spiral arms, a merger",
    "dust_lane":
    "sdss, elliptical galaxy, dust lane",
    "ring":
    "sdss, ring galaxy",
    "candels_smooth":
    "candels, smooth, completely round",
    "candels_merger":
    "candels, smooth, in-between round, merging galaxies",
    "candels_spiral":
    "candels, features or disk-shaped, spiral arms pattern, tightly wound spiral arms, obvious bulge prominence",
    "candels_clump":
    "candels, features or disk-shaped, mostly clumpy appearance, there are five or more clumps, the clumps appear in cluster or irregular, no single brightest clump, clumps are not symmetrical, clumps are embedded",

    "hubble_smooth":
    'hubble, smooth, completely round',
    "hubble_inbetween":
    'smooth, in-between round',
    "hubble_spiral":
    'features or disk-shaped, spiral arms pattern, obvious bulge prominence, medium wound spiral arms, 2 spiral arms',
    "hubble_clump1":
    'features or disk-shaped, something odd, a merger, mostly clumpy appearance, one clump is brightest, brightest clump is at center, the clumps appear in spiral shape, clumps are not symmetrical, clumps are not embedded',
    "hubble_clump2":
    'features or disk-shaped, something odd, a merger, mostly clumpy appearance, no single brightest clump, there are 2 clumps, clumps are not symmetrical, clumps are embedded'

    # Corresponds to the prompt in infer_script_full.sh
}


num_images_per_prompt = 4 # adjust this to the number of images you want to display per prompt
num_prompts = len(prompt_dict.keys()) 

# Get only the prompt names that actually have directories
actual_prompt_names = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
num_prompts = len(actual_prompt_names)

# Create a figure to store all sub-charts
fig, axes = plt.subplots(num_prompts,
                         num_images_per_prompt + 1,
                         figsize=(10, 2 * num_prompts))

for i, prompt_name in enumerate(actual_prompt_names):
    # Get prompt text from prompt_dict if available, otherwise use prompt_name
    prompt_text = prompt_dict.get(prompt_name, prompt_name)

    images_path = f'{output_dir}/{prompt_name}'

    # get all .png or .jpg files in the directory
    img_paths = [
        os.path.join(images_path, f) for f in os.listdir(images_path)
        if f.endswith('.png') or f.endswith('.jpg')
    ]

    images = [mpimg.imread(img_path) for img_path in img_paths]

    for j, img in enumerate(images[:num_images_per_prompt]):
        axes[i, j].imshow(img)
        axes[i, j].axis('off')

    axes[i, num_images_per_prompt].text(0.5,
                    0.3,
                    textwrap.fill(prompt_text.replace(',', ', '),
                                  width=20,
                                  break_long_words=False,
                                  break_on_hyphens=False),
                    fontsize=10,
                    ha='center',
                    wrap=True)
    axes[i, num_images_per_prompt].axis('off')

    # Comment out the line below if you want to keep the original images
    # shutil.rmtree(images_path, ignore_errors=True)

# find a unique name
output_name_tmp = 'summary_{model_steps}_{ii}.png'
ii = 0
output_name = output_name_tmp.format(ii=ii, model_steps=model_steps)
while os.path.exists(f'{output_dir}/{output_name}'):
    ii += 1
    output_name = output_name_tmp.format(ii=ii, model_steps=model_steps)

plt.savefig(f'{output_dir}/{output_name}', bbox_inches='tight')
