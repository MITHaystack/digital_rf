#ifndef INCLUDED_GRDRF_API_H
#define INCLUDED_GRDRF_API_H

#include <gnuradio/attributes.h>

#ifdef gnuradio_drf_EXPORTS
#  define GRDRF_API __GR_ATTR_EXPORT
#else
#  define GRDRF_API __GR_ATTR_IMPORT
#endif

#endif /* INCLUDED_GRDRF_API_H */
