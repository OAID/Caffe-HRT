#ifndef __TESTBED_H__
#define __TESTBED_H__

void init_testbed(void);

void run_test(int reptition, int warm_up,void (*test_func)(void *),void * arg);

void release_testbed(void);

#endif
