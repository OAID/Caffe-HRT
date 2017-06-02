#ifndef AARCH64_PMU_H
#define AARCH64_PMU_H

#include <stdint.h>


#define __stringify_1(x...)	#x
#define __stringify(x...)	__stringify_1(x)

#define read_32bit_sysreg(reg) \
    ({\
       uint32_t val;\
       __asm__ __volatile__ (\
          "mrs %0," __stringify(reg):"=r"(val));\
        val;\
    })


#define read_64bit_sysreg(reg) \
    ({\
       uint64_t val;\
       __asm__ __volatile__ (\
          "mrs %0," __stringify(reg):"=r"(val));\
        val;\
    })

#define write_32bit_sysreg(reg,val) \
    ({\
         __asm__ __volatile__ (\
          "msr " __stringify(reg) " ,%0"::"r"(val));\
      })

#define write_64bit_sysreg(reg,val) write_32bit_sysreg(reg,val)

#define MAX_PROF_POINTS 16

struct pmu_event
{
  int  slot;
  int  event_id;
  char * name;
  uint32_t init_val;
  int enabled;
  char * note;
};

struct  prof_stat
{
   int     prof_seq;
   uint32_t max_val;
   uint32_t min_val;
   uint32_t cur_val;
   uint32_t raw_val;
   uint64_t total_val;
   uint32_t enter_count;
   int  cal_offset;
   int  update_last;
};


struct pmu_event_record
{
  struct pmu_event*  p_evt;
  uint32_t last_val;
  uint32_t base_val;
  struct prof_stat prof_stat[MAX_PROF_POINTS];
};

/* all functions in the group must be called on the same CPU */

extern  void init_pmu_registers(void);
extern void dump_pmu_registers(void);

/* create one event with event_id, return slot number in success */
extern int create_pmu_event(char *name,int event_id, 
                          uint32_t init_val, char * note);

extern void release_pmu_event(int slot);

extern void start_pmu_event(int slot);

extern void stop_pmu_event(int slot);

extern void set_pmu_event_base(int slot);

extern void record_pmu_event(int slot, int seq, int cal_offset, int update_last);

extern void dump_pmu_event_stat(int slot);

extern struct pmu_event_record *  get_pmu_event_record(int slot);

extern uint32_t get_pmu_stat_avg(int slot); /* adding all phase avg together */

/* regsiter level interface */

extern int setup_event_counter(int slot, int event_id);

static inline void start_pmu_counter(int slot)
{
           uint32_t mask=1<<slot;

           write_32bit_sysreg(PMCNTENSET_EL0,mask);
}

static inline void stop_pmu_counter(int slot)
{
   uint32_t mask=1<<slot;

   write_32bit_sysreg(PMCNTENCLR_EL0,mask);
}


static inline void write_pmu_counter(int slot,uint32_t val)
{
   write_32bit_sysreg(PMSELR_EL0,slot);

   if(slot<31)
      write_32bit_sysreg(PMXEVCNTR_EL0, val);
   else
      write_64bit_sysreg(PMXEVCNTR_EL0,val);

}

static inline uint32_t read_pmu_counter(int slot)
{
   write_32bit_sysreg(PMSELR_EL0,slot);
   return read_32bit_sysreg(PMXEVCNTR_EL0);
}

#endif
