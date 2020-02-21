API Documentation
=================

.. automodule:: spaudiopy



Multiprocessing
---------------
If the function has an argument called `jobs_count` the implementation allows launching multiple processing jobs.
Keep in mind that there is always a certain computational overhead involved, and more jobs is not always faster.

Especially on Windows, you then have to protect your `main()` function with::

    if __name__ == '__main__':
        # Put your code here
