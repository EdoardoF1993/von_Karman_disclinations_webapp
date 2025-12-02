# Disclinations

### Prerequisites

The project assumes basic knowledge of the theory of infinitesimal elasticity and finite element methods.

Basic knowledge of Python will be assumed, see https://github.com/jakevdp/WhirlwindTourOfPython
to brush up if you feel unsure.

Basic knowledge of git as a versioning system with feature-branch workflow
https://gist.github.com/brandon1024/14b5f9fcfd982658d01811ee3045ff1e

Remember to set your name and email before pushing to the repository,
either locally or globally, see https://www.phpspiderblog.com/how-to-configure-username-and-email-in-git/

### Weekly updates (merge from main)
```
git checkout main
git pull
git checkout yourname-branch
git merge main
```

### To run the code (on Docker)
On Linux, to build the docker image

```
docker build -t myapp .
```

To run the docker image
```
docker run -p 10000:10000 myapp
```

### Authors
- Edoardo Fabbrini
  
