#include <stdio.h>
#include <stdlib.h>

#include "pmu.h"

#define MAX_SLOT_NUM 		64 	/* defined by ARMv8 SPEC*/
#define ARMV8_PMCR_N_SHIFT      11      /* Number of counters supported */                             
#define ARMV8_PMCR_N_MASK       0x1f                                                                    


static __thread  struct pmu_event_record * g_rec_ptr[MAX_SLOT_NUM];
static __thread  int max_counter_slot;

/* start and stop counter */

static void stop_event_profile(struct pmu_event_record * p_record)
{
	int slot=p_record->p_evt->slot;

	p_record->p_evt->enabled=0;

	stop_pmu_counter(slot);
}


static void init_pmu_event_record(struct pmu_event * p_evt, struct pmu_event_record * p_record)
{
	struct prof_stat *p_stat;
	int i=0;

	p_record->p_evt=p_evt;
	p_record->last_val=p_evt->init_val;
	p_record->base_val=p_evt->init_val;
	p_stat=p_record->prof_stat;

	for(i=0;i<MAX_PROF_POINTS;i++)
	{
		p_stat[i].prof_seq=i;
		p_stat[i].max_val=0;
		p_stat[i].min_val=-1U;
		p_stat[i].raw_val=0xdeadbeaf;
		p_stat[i].cur_val=0;
		p_stat[i].total_val=0;
		p_stat[i].enter_count=0;         
	}
}

static void start_event_profile(struct pmu_event_record * p_record)
{
	int slot=p_record->p_evt->slot;
	struct prof_stat *p_stat;
        int i;

	p_record->p_evt->enabled=1;

	p_stat=p_record->prof_stat;

	for(i=0;i<MAX_PROF_POINTS;i++)
	{
		p_stat[i].prof_seq=i;
		p_stat[i].max_val=0;
		p_stat[i].min_val=-1U;
		p_stat[i].raw_val=0xdeadbeaf;
		p_stat[i].cur_val=0;
		p_stat[i].total_val=0;
		p_stat[i].enter_count=0;         
        }

	write_pmu_counter(slot,p_record->p_evt->init_val);
	start_pmu_counter(slot);

}

/* create event and profile */


int setup_event_counter(int slot, int event_id)
{

	if(slot==31)
		return 0;

	if(event_id>1023)
		return -1;

	write_32bit_sysreg(PMSELR_EL0,slot);
	write_32bit_sysreg(PMXEVTYPER_EL0,event_id);

	return 0;
}


static struct pmu_event_record * create_pmu_event_record(char *name, int slot, 
		int event_id, uint32_t init_val, char * note)
{
	struct pmu_event * p_evt;
	struct pmu_event_record * p_record;

	if(setup_event_counter(slot,event_id)<0)
		return NULL;

	p_evt=malloc(sizeof(struct pmu_event));

	if(p_evt==NULL)
		return NULL;

	p_evt->name=name;
	p_evt->slot=slot;
	p_evt->event_id=event_id;
	p_evt->init_val=init_val;
	p_evt->note=note;
	p_evt->enabled=0;

	p_record=malloc(sizeof(struct pmu_event_record));

	if(p_record==NULL)
	{
		free(p_evt);
		return NULL;
	}

	p_record->p_evt=p_evt;

	init_pmu_event_record(p_evt,p_record);

	return p_record;
}


static void record_event_prof(struct pmu_event_record * p_record, 
		int prof_seq, int cal_offset, int update_last)
{
	struct prof_stat * p_stat;
	uint32_t evt_val;

	evt_val=read_pmu_counter(p_record->p_evt->slot);

	p_stat=&p_record->prof_stat[prof_seq];

	p_stat->cal_offset=cal_offset;
	p_stat->update_last=update_last;
	p_stat->raw_val=evt_val;

	if(cal_offset)
		p_stat->cur_val=evt_val-p_record->last_val;
	else
		p_stat->cur_val=evt_val-p_record->base_val;

	if(update_last)
		p_record->last_val=evt_val;

	p_stat->total_val+=p_stat->cur_val;

	if(p_stat->cur_val>p_stat->max_val)
		p_stat->max_val=p_stat->cur_val;

	if(p_stat->cur_val<p_stat->min_val)
		p_stat->min_val=p_stat->cur_val;

	p_stat->enter_count++;
}


static void release_pmu_event_record(struct pmu_event_record * p_record)
{
     struct pmu_event * p_evt;

     p_evt=p_record->p_evt;

     if(p_evt->enabled)
         stop_pmu_counter(p_evt->slot);

     free(p_evt);
     free(p_record);
}




/* debugging */

static void dump_pmu_event(struct pmu_event * p_evt)
{
        
	printf("event[%s/0x%x]: slot [%d] init_val[0x%x] enabled[%d]",
			p_evt->name,p_evt->event_id,p_evt->slot,p_evt->init_val,
			p_evt->enabled);

	if(p_evt->note)
		printf(" note[%s]\n",p_evt->note);
	else
		printf("\n");
}

static void dump_pmu_event_record(struct pmu_event_record * p_record)
{
	int i;
	struct prof_stat * p_stat;
        uint64_t total_avg_val=0;
        int count=0;
        uint32_t avg;

        printf("------------------------------------------------------------------------\n");

	dump_pmu_event(p_record->p_evt);

	p_stat=&p_record->prof_stat[0];

	for(i=0;i<MAX_PROF_POINTS;i++)
	{
		if(p_stat[i].enter_count==0)
			continue;

                avg=(uint32_t)(p_stat[i].total_val/p_stat[i].enter_count);

		printf("stat [%d]: max/min/avg [0x%x/0x%x/0x%x] total [0x%lx] count[%u]\n",
				i,p_stat[i].max_val,p_stat[i].min_val,
				avg,
				p_stat[i].total_val,p_stat[i].enter_count);
		printf("         raw_val[0x%x] cal_offset[%d] update_last[%d]\n",
				p_stat[i].raw_val,p_stat[i].cal_offset,p_stat[i].update_last);

                count++;

                total_avg_val+=avg;
 

	}

        printf("total [%d] points, the sum of average number is: [0x%lx]\n\n",count,total_avg_val);
}

/* output interface */

void init_pmu_registers(void)
{
	/* enabled PMU in PMCR*/
	write_32bit_sysreg(PMCR_EL0,0x1);
	max_counter_slot=(read_32bit_sysreg(PMCR_EL0) >> ARMV8_PMCR_N_SHIFT)&ARMV8_PMCR_N_MASK;
}


#define dump_32bit_sysreg(reg) \
	printf(__stringify(reg) " is [0x%08x]\n",read_32bit_sysreg(reg))

#define dump_64bit_sysreg(reg) \
	printf(__stringify(reg) " is [0x%016llx]\n",read_32bit_sysreg(reg))


void dump_pmu_registers(void)
{
	dump_32bit_sysreg(PMCEID0_EL0);
	dump_32bit_sysreg(PMCEID1_EL0);
	dump_32bit_sysreg(PMOVSSET_EL0);
	dump_32bit_sysreg(PMCR_EL0);
	dump_32bit_sysreg(PMUSERENR_EL0);
        dump_32bit_sysreg(PMCNTENSET_EL0);
}

struct pmu_event_record *  get_pmu_event_record(int slot)
{
	return  g_rec_ptr[slot];
}

int create_pmu_event(char *name,int event_id,
		uint32_t init_val, char * note)
{
	int i;

	struct pmu_event_record * p_record;

	for(i=0;i<max_counter_slot;i++)
	{
		if(g_rec_ptr[i]==NULL)
			break;
	}

	if(i==max_counter_slot)
		return -1;

	p_record=create_pmu_event_record(name,i,event_id,init_val,note);

	if(p_record==NULL)
		return -1;

	g_rec_ptr[i]=p_record;

	return i;
}

void release_pmu_event(int slot)
{
	struct pmu_event_record * p_record;

	p_record=g_rec_ptr[slot];

	if(p_record)
		release_pmu_event_record(p_record);

	g_rec_ptr[slot]=NULL;
}

void start_pmu_event(int slot)
{
	struct pmu_event_record * p_record;

	p_record=g_rec_ptr[slot];

	start_event_profile(p_record);
}

void stop_pmu_event(int slot)
{
	struct pmu_event_record * p_record;

	p_record=g_rec_ptr[slot];

	stop_event_profile(p_record);
}

void record_pmu_event(int slot, int seq, int cal_offset, int update_last)
{
	struct pmu_event_record * p_record;

	p_record=g_rec_ptr[slot];

	record_event_prof(p_record,seq,cal_offset,update_last);
}

void dump_pmu_event_stat(int slot)
{
	struct pmu_event_record * p_record;

	p_record=g_rec_ptr[slot];

	dump_pmu_event_record(p_record);
}


uint32_t get_pmu_stat_avg(int slot)
{
	struct pmu_event_record * p_record;
        struct prof_stat * p_stat;
        uint32_t total_avg=0;
        uint32_t avg;
        int i;

	p_record=g_rec_ptr[slot];

        for(i=0;i<MAX_PROF_POINTS;i++)
        {
           p_stat=&p_record->prof_stat[i];

           if(p_stat->enter_count==0)
                 continue;
            avg=p_stat->total_val/p_stat->enter_count;
            total_avg+=avg;
        }
 
        return total_avg;
}

void set_pmu_event_base(int slot)
{
      struct pmu_event_record * p_record;

      uint32_t val;

      p_record=g_rec_ptr[slot];
 
      val=read_pmu_counter(slot);

       p_record->last_val=val;
       p_record->base_val=val;

}
