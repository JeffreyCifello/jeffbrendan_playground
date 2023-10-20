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

