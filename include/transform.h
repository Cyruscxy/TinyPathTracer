#pragma once

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "math/quat.h"
#include <memory>

class Transform
{
public:
	Transform() = default;
	Transform(Vec3 loc, Vec3 angle, Vec3 s) : m_location(loc), m_rotation(angle), m_scale(s) {}
	Transform(Vec3 loc, Quat quat, Vec3 s) : m_location(loc), m_rotation(quat), m_scale(s) {}

	Mat4 localToWorld()
	{
		if ( std::shared_ptr<Transform> parent = m_parent.lock() )
		{
			return parent->localToWorld() * localToParent();
		}
		else
		{
			return localToParent();
		}
	}

	Mat4 localToParent()
	{
		return Mat4::Translate(m_location) * 
			Mat4(Quat::RotateFromQuat(m_rotation)) * 
			Mat4::Scale(m_scale);
	}

	Mat4 worldToLocal()
	{
		if ( std::shared_ptr<Transform> parent = m_parent.lock() )
		{
			return parentToLocal() * parent->worldToLocal();
		}
		else
		{
			return parentToLocal();
		}
	}

	Mat4 parentToLocal()
	{
		return Mat4::Scale(m_scale.reciprocal()) *
			Quat::RotateFromQuat(m_rotation.inverse()) *
			Mat4::Translate(-m_location);
	}

private:
	std::weak_ptr<Transform> m_parent;
	Vec3 m_location;
	Quat m_rotation;
	Vec3 m_scale;
};

#endif
