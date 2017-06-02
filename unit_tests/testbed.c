#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>


#include  "pmu.h"
#include "testbed.h"

struct armv8_event
{
	char * name;
	int id;
	uint32_t init_val;
	char * note;
};


static struct armv8_event a57_list[6]=
{
	{"INST",0x8,0,"instruction retired"},
	{"CYCL",0x11,0,"CPU running cycle"},
	{"L1D MISS",0x3,0,"L1D CACHE MISS/REFILL"},
	{"L1D ACCESS",0x4,0,"L1D CACHE ACCESS"},
	{"L2 MISS",0x17,0,"L2 CACHE MISS/REFILL"},
	{"L2 ACCESS",0x16,0,"L2 CACHE ACCESS"}
};

static int e[6];

void init_testbed(void)
{
        int i;
        struct armv8_event * p_list;

	init_pmu_registers();

	p_list=a57_list;

	for(i=0;i<6;i++)
	{
		e[i]=create_pmu_event(p_list[i].name,p_list[i].id,
				p_list[i].init_val,p_list[i].note);
	}

}

void run_test(int reptition, int warm_up,void (*test_func)(void *),void * arg)
{
        uint32_t t0,t1;
	uint32_t freq;
	uint32_t cycle;
	uint64_t total_time=0;
	uint32_t loop_count=0;
	int i;
     
        if(warm_up)
           test_func(arg);
         

	freq=read_32bit_sysreg(CNTFRQ_EL0);

	start_pmu_event(e[0]);
	start_pmu_event(e[1]);
	start_pmu_event(e[2]);
	start_pmu_event(e[3]);
	start_pmu_event(e[4]);
	start_pmu_event(e[5]);

	set_pmu_event_base(e[0]);
	set_pmu_event_base(e[1]);
	set_pmu_event_base(e[2]);
	set_pmu_event_base(e[3]);
	set_pmu_event_base(e[4]);
	set_pmu_event_base(e[5]);

	t0=read_32bit_sysreg(CNTVCT_EL0);

	for(i=0;i<reptition;i++)
	{
	    test_func(arg);

	record_pmu_event(e[0],0,1,1);
	record_pmu_event(e[1],0,1,1);
	record_pmu_event(e[2],0,1,1);
	record_pmu_event(e[3],0,1,1);
	record_pmu_event(e[4],0,1,1);
	record_pmu_event(e[5],0,1,1);

	t1=read_32bit_sysreg(CNTVCT_EL0);
	loop_count++;
	total_time+=(t1-t0);
        t0=t1;

	}
        

	stop_pmu_event(e[0]);
	stop_pmu_event(e[1]);
	stop_pmu_event(e[2]);
	stop_pmu_event(e[3]);
	stop_pmu_event(e[4]);
	stop_pmu_event(e[5]);

	dump_pmu_event_stat(e[0]);
	dump_pmu_event_stat(e[1]);
	dump_pmu_event_stat(e[2]);
	dump_pmu_event_stat(e[3]);
	dump_pmu_event_stat(e[4]);
	dump_pmu_event_stat(e[5]);


	printf("\n------------------------------------\n\n");


	cycle=get_pmu_stat_avg(e[1]);
	t0=total_time/loop_count;


	printf("freq is 0x%x\n",freq);
	printf("pysical counter pass: 0x%x (0x%lx/%u)\n",t0,total_time,loop_count);
	printf("coverted to ms: %.3f\n",1000.0*t0/freq);


	printf("CPU freq: %.2f MHZ (cycle:0x%x)\n",(float)freq*cycle/t0/1000000,cycle);

	printf("IPC is: %.2f \n",(float)get_pmu_stat_avg(e[0])/cycle);
	printf("L1 CACHE MISS  is: %.2f \n",(float)get_pmu_stat_avg(e[2])/get_pmu_stat_avg(e[3]));
	printf("L2 CACHE MISS  is: %.2f \n",(float)get_pmu_stat_avg(e[4])/get_pmu_stat_avg(e[5]));

        /*reset all record */

}

void release_testbed(void)
{

	release_pmu_event(e[0]);
	release_pmu_event(e[1]);
	release_pmu_event(e[2]);
	release_pmu_event(e[3]);
	release_pmu_event(e[4]);
	release_pmu_event(e[5]);
}
