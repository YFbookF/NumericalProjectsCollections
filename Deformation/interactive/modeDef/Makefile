
BIN = ../bin/
SRC = ./

# CC should be set to the name of your favorite C++ compiler.

CC = gcc

RM = /bin/rm




##########################################################################
### Files and Paths - this is probably the only section you'll need to change
##########################################################################


# set this to the glut path
GLUT		= $(HOME)/glut-3.7

# Numerical Recipes SVD routine: either set NRDIR to the correct
# directory, or set NRFILES to whatever files you'd like to include
# to define the svdcmp routine. 

# set this to the Numerical Recipies dir 
NRDIR		= /home/eecs/job/Code/Misc/NRecipies/Src

# set this to the Numerical Recipies files you need (svdcmp.c, nrutil.c, pythag.c)
#NRFILES         = $(NRDIR)/svdcmp.c $(NRDIR)/nrutil.c $(NRDIR)/pythag.c
NRFILES         = 

##########################################################################
### The rest...
##########################################################################



# The source files for the project.
# get all cpp source files
SRCS		+= $(wildcard *.cpp)
# get all c source files
SRCS		+= $(wildcard *.c)
SRCS		+= $(wildcard *.C)
#SRCS		+= $(foreach DIR,$(SRCDIRS),$(subst $(DIR)/,,$(wildcard $(DIR)/*.cpp)))
#SRCS		+= $(foreach DIR,$(SRCDIRS),$(subst $(DIR)/,,$(wildcard $(DIR)/*.c)))
#SRCS		+= $(foreach DIR,$(SRCDIRS),$(subst $(DIR)/,,$(wildcard $(DIR)/*.C)))
SRCS		+= $(wildcard simulation/*.cpp)
SRCS		+= $(wildcard shared/*.cpp)
SRCS		+= $(wildcard modal/*.cpp)
SRCS		+= $(wildcard math/*.cpp)

# Additional objects to link. Only add things that aren't built from SRCS!
OBJS 		+= 

# Additional libs to link with.
LIBS		+= GL GLU glut stdc++



# Additional locations for header files
#INCDIRS		+= $(GL)/include/
INCDIRS		+= $(GLUT)/include
INCDIRS		+= .
INCDIRS		+= modal
INCDIRS		+= shared

#use CLAPACK for SVD stuff
#DEFINES += USE_CLAPACK
#LIBS 	+= lapack_LINUX blas_LINUX F77 I77
#INCDIRS += $(HOME)/CLAPACK $(HOME)/CLAPACK/F2CLIBS $(HOME)/CLAPACK/BLAS/WRAP
#LIBDIRS += $(HOME)/CLAPACK $(HOME)/CLAPACK/F2CLIBS

# Additional locations for library files
LIBDIRS			+= $(GLUT)/lib/glut
LIBDIRS			+= .

# Additional locations for source files
SRCDIRS		= simulation shared math modal


#james' spring mass system thing
#SM_DIR = /project/eecs/anim_dsk11/job/Code/Systems/SpringMass/
#SM_INC = -I$(SM_DIR)/Inc
#SM_SRC = $(SM_DIR)/Src
#SM_FILES = smBounds.C smRigidBodies.C smMatrix.C smVector.C smNormals.C smXform.C smRay.C smRotations.C
#SM_SRCS	= $(addprefix $(SM_SRC)/, $(SM_FILES)) 
#SM_LIB = -lSm

DEFINES += MATH_DOUBLE

# The action starts here.

#INCDIRS += /usr/lib/gcc-lib/i386-redhat-linux/2.96/include/ .

all: simulate

.cpp.o: 
	$(CC) $(CSWITCHES) -c $<

simulate: $(OBJS)
	$(CC) $(CSWITCHES) $(SRCS) $(addprefix -l, $(LIBS)) $(addprefix -L, $(LIBDIRS)) $(addprefix -I, $(INCDIRS)) $(NRFILES) $(addprefix -D, $(DEFINES)) -o simulate

deform:

clean:
	$(RM) $(SRC)*.o
