# Finite Element Methods in Computational Mechanics

Welcome to our repo for weekly assignments in the UiO course 
[MEK9250/MEK4250](https://www.uio.no/studier/emner/matnat/math/MEK4250/index-eng.html). 
This repo is a collection of solutions from different students from the spring semester of 2023, misusing branches in git. 
Each student should have its own branch containing their solution.

Feel free to look at the solution of others, discussion is always encouraged. 
However, please do not make any changes to branches other than your own.


## Getting started

This part is mainly for those who are not too familiar with git and GitHub.

Start by cloning the repo using the following command in your terminal:
```
git clone git@github.com:eirillsh/MEK9250.git
```
Make sure you have set up [ssh](https://docs.github.com/en/authentication/connecting-to-github-with-ssh). Remote should be automatically set.

Create (`-c`) and switch to your personal branch:
```
git -c swtich <branch>
```
where `<branch>` should be the name of your branch.

Move your existing work to the folder add, commit, and push all at once:
```
git add .
git commit -m <commit-message>
git push -u origin <branch>
```
Using `-u` will set up-stream to this GitHub repo, specifically to your branch. 
After this initial push,  use `git push` to push local changes to GitHub
and `git pull` to pull changes on GitHub to your local repo.


## Resources

Links for 2023:
- [Course web site](https://www.uio.no/studier/emner/matnat/math/MEK4250/v23/index.htm): course schedule.
- [Info](https://kent-and.github.io/mek4250/2023/index.html): Kent's info page.
- [Book](https://kent-and.github.io/mek4250/2023/book_jan23.pdf): Kent's lecture notes (might be updated during the course).
- [FAFEM](https://www.simula.no/education/courses/faefem-functional-analysis-essentials-finite-element-method): Simula crash course in functional analysis.


## Environments

There are several options regarding suitable libraries implementing the finite element method. Some options are listed below. 

It might be beneficial to create a separate 
[conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) 
for your FEM calculations. 
From my (Eirill's) experience, the FEM libraries (FEniCS) are at times incompatible with existing packages in the environment. 
Keeping them separate will ensure that you do not break your *regular* environment. 
This is especially important for FEniCS, as it has not been updated in a few years. 
Installing FEniCS in your default environment might therefore require downgrading some of your existing packages.

### FEniCS

The latest (official) version of  FEniCS is 2019.1.0, see 
[documentation](https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/) 
for details.

The file `fenics-env.yml` contains a minimal conda environment for `fenics`. 
Create this environment by running the following command line in your terminal:
```
conda env create -f fenics-env.yml
```
To activate the new environment use:
```
conda activate fenics
```
or, if your conda is very old:
```
source activate fenics
```

### FEniCSx

...
