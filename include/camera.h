#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "transform.h"
#include <cmath>

class Camera
{
public:
	Camera() : m_verticalFOV(60.0f), m_aspectRatio(1.77778f), m_nearPlane(0.1f)
	{
		if (!m_transform) m_transform = std::make_shared<Transform>(Vec3(0.0f), Vec3(0.0f), Vec3(1.0f));
	}
	Camera(Real fov, Real asp, Real nearZ) : m_verticalFOV(fov), m_aspectRatio(asp), m_nearPlane(nearZ)
	{
		if (!m_transform) m_transform = std::make_shared<Transform>(Vec3(0.0f), Vec3(0.0f), Vec3(1.0f));
	}
	Camera(Real fov, Real asp, Real nearZ, const Transform& trans) : m_verticalFOV(fov), m_aspectRatio(asp), m_nearPlane(nearZ)
	{
		m_transform = std::make_shared<Transform>(trans);
	}
	Camera(Real fov, Real asp, Real nearZ, Transform&& trans) : m_verticalFOV(fov), m_aspectRatio(asp), m_nearPlane(nearZ)
	{
		m_transform = std::make_shared<Transform>(std::forward<Transform>(trans));
	}
	Camera(Vec3 loc, Vec3 angle, Vec3 s) : Camera()
	{
		m_transform = std::make_shared<Transform>(loc, angle, s);
	}
	Camera(Vec3 loc, Quat q, Vec3 s) : Camera()
	{
		m_transform = std::make_shared<Transform>(loc, q, s);
	}
	Camera(Real fov, Real asp, Real nearZ, Vec3 loc, Vec3 angle, Vec3 s) : 
		m_transform(std::make_shared<Transform>(loc, angle, s)) ,
		m_verticalFOV(fov), m_aspectRatio(asp), m_nearPlane(nearZ){ }
	Camera(Real fov, Real asp, Real nearZ, Vec3 loc, Quat q, Vec3 s) :
		m_transform(std::make_shared<Transform>(loc, q, s)),
		m_verticalFOV(fov), m_aspectRatio(asp), m_nearPlane(nearZ) { }

	[[nodiscard]]
	Mat4 projection() const
	{
		float f = 1.0f / std::tan(m_verticalFOV * MathConst::Rad * 0.5f);
		auto& n = m_nearPlane;
		auto& ar = m_aspectRatio;
		return Mat4(
			Vec4(f/ar, 0.0f,      0.0f,  0.0f),
			Vec4(0.0f,    f,      0.0f,  0.0f),
			Vec4(0.0f, 0.0f,     -1.0f, -1.0f),
			Vec4(0.0f, 0.0f, -2.0f * n,  0.0f));
	}

	[[nodiscard]] Real getVFov() const { return m_verticalFOV; }
	[[nodiscard]] Real getAspRatio() const { return m_aspectRatio; }
	[[nodiscard]] Real getNearPlane() const { return m_nearPlane; }

	std::shared_ptr<Transform> m_transform;
private:

	Real m_verticalFOV;
	Real m_aspectRatio;
	Real m_nearPlane;
};

#endif