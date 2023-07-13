#include "path_tracer.h"
#include "pathtrace.cuh"

PathTracer::PathTracer() :
m_camera(),
m_displayer(),
m_width(m_displayer.m_windowExtent.width),
m_height(m_displayer.m_windowExtent.height),
m_rays((size_t)m_width * m_height)
{
	
}

void PathTracer::doTrace()
{
	
}

void PathTracer::sampleRay()
{
	
}
