# jeffbrendan_playground
Here is my test text for this repo. I made a clone of this repo by opening WSL and running this:
```
git clone https://github.com/JeffreyCifello/jeffbrendan_playground
```

I am going to make a new branch and move to that branch with:
```
git branch new_branch
git checkout new_branch
```
I will create a new file using `vim`, and edit it a bit. Enter "insert" mode with `i` and save changes by pressing ESCAPE then `:wq`. 
```
vim test_file_jeff.txt
```

I will add these changes to the tracked changes. I will commit the changes accompanied by a message. I will push these to the original repository for review:
```
git add .
git commit -m "This is a test commit by jeff."
git push origin new_branch
```

Edit: this didn't totally work. Before being able to commit, I had to create a personal access token. Information is here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens.  
Navigate to the [creation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token) section.

I followed the instructions and created a fine-grained key for 6 months use. I checked the top 4 or so Repository Permission because a lot of them seemed like features I wouldn't use. I gave myself the ability to block other accounts. 
I copied and saved the key **OUTSIDE** of the repository. 


Additionally, before doing the commit and push, I explicitly set my username with:
```
git config --global user.name "JeffreyCifello"
```

Edit 2: When doing the push to origin, I used JeffreyCifello as my user name and my recently generated key as my password. 
