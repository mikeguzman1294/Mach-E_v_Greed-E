# Mach-E_v_Greed-E
Python game to instruct on Artificial Intelligence Subjects.


PS: You will need to have pygame installed in your machine, open a terminal and run:

<pre>pip install pygame</pre> 

In the context of the AI course, we are going to simplify the rules of PyRat a bit.
In fact, we are going to remove all walls and mud penalties. Also, we are not going to consider symmetric mazes anymore.

As such, a default game is launched with the following parameters. Please try now (note that you may have to type python instead of python3): 

<pre>python pyrat.py -p 40 -md 0 -d 0 --nonsymmetric</pre>

An empty labyrinth will appear.

Please check out all the options offered by the pyrat software, by running : 

<pre>python pyrat.py -h</pre>

Importantly, there are options to change the size of the map, the number of battery, which will be very useful later to benchmark your own solutions. 

In the supervised and unsupervised projects, we are going to look at plays between two greedy algorithms. Generating 1000 such games while saving data is easily obtained with PyRat. 

Open another terminal to launch the next command line. Generating 1000 games will take a few minutes.

<pre>python pyrat.py --width 21 --height 15 -p 40 -md 0 -d 0 --nonsymmetric --rat AIs/manh.py --python AIs/manh.py --tests 1000 --nodrawing --synchronous --save</pre>

The 1000 generated games will be in the "saves" folder. Each time you execute the command new games are added to the saves folder. You have to manually delete the old games if you do not want to use them (for example, if you change the size of the labyrinth or if you want to train your IA on new games).

As bonus, to run a cute visual simulation to understand Pyrat with the players controlled by the greedy approach AI you can run the following command:

<pre>python pyrat.py --width 20 --height 20 -p 40 -md 0 -d 0 --nonsymmetric --rat AIs/manh.py --python AIs/manh.py</pre>
