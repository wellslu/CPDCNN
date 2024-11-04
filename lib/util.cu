#include <sys/time.h>
double get_time() {
   struct timeval t;
   gettimeofday(&t, NULL);
   return  t.tv_sec + t.tv_usec/1000000.0;
}