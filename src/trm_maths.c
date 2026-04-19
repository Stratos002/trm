#include "trm_maths.h"

#include <math.h>
#include <stdlib.h> 
#include <stdint.h>

// === Matrix4x4 ===

void TRM_Matrix4x4_transpose(struct TRM_Matrix4x4* pResult)
{
	for(uint32_t row = 0; row < 4; ++row)
	{
		for(uint32_t column = row + 1; column < 4; ++column)
		{
			const float tmp = pResult->m[row][column];
			pResult->m[row][column] = pResult->m[column][row];
			pResult->m[column][row] = tmp;
		}
	}
}

void TRM_Matrix4x4_getUpperLeft(struct TRM_Matrix4x4 matrix, struct TRM_Matrix3x3* pResult)
{
	for(uint32_t row = 0; row < 3; ++row)
	{
		for(uint32_t column = 0; column < 3; ++column)
		{
			pResult->m[row][column] = matrix.m[row][column];
		}
	}
}

void TRM_Matrix4x4_multiplyWithMatrix4x4(struct TRM_Matrix4x4 matrix0, struct TRM_Matrix4x4 matrix1, struct TRM_Matrix4x4* pResult)
{
	for(uint32_t row = 0; row < 4; ++row)
	{
		for(uint32_t column = 0; column < 4; ++column)
		{
			pResult->m[row][column] = 0.0f;
			for(uint32_t k = 0; k < 4; ++k)
			{
				pResult->m[row][column] += matrix0.m[row][k] * matrix1.m[k][column];
			}
		}
	}
}

void TRM_Matrix4x4_multiplyWithVector4(struct TRM_Matrix4x4 matrix, struct TRM_Vector4 vector, struct TRM_Vector4* pResult)
{
	pResult->x = matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y + matrix.m[0][2] * vector.z + matrix.m[0][3] * vector.w;
	pResult->y = matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y + matrix.m[1][2] * vector.z + matrix.m[1][3] * vector.w;
	pResult->z = matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y + matrix.m[2][2] * vector.z + matrix.m[2][3] * vector.w;
	pResult->w = matrix.m[3][0] * vector.x + matrix.m[3][1] * vector.y + matrix.m[3][2] * vector.z + matrix.m[3][3] * vector.w;
}

void TRM_Matrix4x4_getIdentity(struct TRM_Matrix4x4* pResult)
{
	for(uint32_t i = 0; i < 4; ++i)
	{
		pResult->m[i][i] = 1.0f;
	}
}

void TRM_Matrix4x4_getScaling(float scalingX, float scalingY, float scalingZ, struct TRM_Matrix4x4* pResult)
{
	pResult->m[0][0] = scalingX;
	pResult->m[1][1] = scalingY;
	pResult->m[2][2] = scalingZ;
	pResult->m[3][3] = 1.0f;
}

void TRM_Matrix4x4_getRotation(float yaw, float pitch, float roll, struct TRM_Matrix4x4* pResult)
{
	// idea : A * B = take the world space transform made by B and apply A -> new transform in world space
	// hence roll then pitch then yaw (yaw * pitch * roll)
	// it wouldn't work otherwise or would "deform" the expected rotations in weird ways

	const float cosy = cosf(yaw);
	const float siny = sinf(yaw);
	const float cosp = cosf(pitch);
	const float sinp = sinf(pitch);
	const float cosr = cosf(roll);
	const float sinr = sinf(roll);

	pResult->m[0][0] = sinp * sinr * siny + cosr * cosy;
	pResult->m[0][1] = cosr * sinp * siny - cosy * sinr;
	pResult->m[0][2] = -cosp * siny;
	pResult->m[0][3] = 0.0f;

	pResult->m[1][0] = cosp * sinr;
	pResult->m[1][1] = cosp * cosr;
	pResult->m[1][2] = sinp;
	pResult->m[1][3] = 0.0f;

	pResult->m[2][0] = -cosy * sinp * sinr + cosr * siny;
	pResult->m[2][1] = -cosr * cosy * sinp - sinr * siny;
	pResult->m[2][2] = cosp * cosy;
	pResult->m[2][3] = 0.0f;

	pResult->m[3][0] = 0.0f;
	pResult->m[3][1] = 0.0f;
	pResult->m[3][2] = 0.0f;
	pResult->m[3][3] = 1.0f;
}

void TRM_Matrix4x4_getInverseRotation(float yaw, float pitch, float roll, struct TRM_Matrix4x4* pResult)
{
	// same idea as rotation, but we need to "undo" each transformation/rotation
	// so we apply each rotation in reverse yaw then pitch then roll (roll * pitch * yaw) but with a negative angle

	const float cosy = cosf(-yaw);
	const float siny = sinf(-yaw);
	const float cosp = cosf(-pitch);
	const float sinp = sinf(-pitch);
	const float cosr = cosf(-roll);
	const float sinr = sinf(-roll);

	pResult->m[0][0] = -sinp * sinr * siny + cosr * cosy;
	pResult->m[0][1] = -cosp * sinr;
	pResult->m[0][2] = -cosy * sinp * sinr - cosr * siny;
	pResult->m[0][3] = 0.0f;

	pResult->m[1][0] = cosr * sinp * siny + cosy * sinr;
	pResult->m[1][1] = cosp * cosr;
	pResult->m[1][2] = cosr * cosy * sinp - sinr * siny;
	pResult->m[1][3] = 0.0f;

	pResult->m[2][0] = cosp * siny;
	pResult->m[2][1] = -sinp;
	pResult->m[2][2] = cosp * cosy;
	pResult->m[2][3] = 0.0f;

	pResult->m[3][0] = 0.0f;
	pResult->m[3][1] = 0.0f;
	pResult->m[3][2] = 0.0f;
	pResult->m[3][3] = 1.0f;
}

void TRM_Matrix4x4_getTranslation(struct TRM_Vector3 position, struct TRM_Matrix4x4* pResult)
{
	pResult->m[0][3] = position.x;
	pResult->m[1][3] = position.y;
	pResult->m[2][3] = position.z;
}

void TRM_Matrix4x4_getView(struct TRM_Vector3 position, float yaw, float pitch, float roll, struct TRM_Matrix4x4* pResult)
{
	TRM_Matrix4x4_getInverseRotation(yaw, pitch, roll, pResult);

	pResult->m[3][0] = -position.x;
	pResult->m[3][1] = -position.y;
	pResult->m[3][2] = -position.z;
}

void TRM_Matrix4x4_getProjection(float fov, float aspectRatio, struct TRM_Matrix4x4* pResult)
{
	const float invTanHalfFov = 1.0f / tanf(fov / 2.0f);

	const float zNear = 1.0f;
	//const float zFar = 10000.0f;

	pResult->m[0][0] = invTanHalfFov / aspectRatio;
	pResult->m[1][1] = invTanHalfFov;
	pResult->m[2][2] = 1.0f;
	pResult->m[2][3] = -zNear;
	pResult->m[3][2] = 1.0f;
}

// === Matrix3x3 ===

void TRM_Matrix3x3_multiplyWithMatrix3x3(struct TRM_Matrix3x3 matrix0, struct TRM_Matrix3x3 matrix1, struct TRM_Matrix3x3* pResult)
{
	for(uint32_t row = 0; row < 3; ++row)
	{
		for(uint32_t column = 0; column < 3; ++column)
		{
			pResult->m[row][column] = 0.0f;
			for(uint32_t k = 0; k < 3; ++k)
			{
				pResult->m[row][column] += matrix0.m[row][k] * matrix1.m[k][column];
			}
		}
	}
}

void TRM_Matrix3x3_multiplyWithVector3(struct TRM_Matrix3x3 matrix, struct TRM_Vector3 vector, struct TRM_Vector3* pResult)
{
	pResult->x = matrix.m[0][0] * vector.x + matrix.m[0][1] * vector.y + matrix.m[0][2] * vector.z;
	pResult->y = matrix.m[1][0] * vector.x + matrix.m[1][1] * vector.y + matrix.m[1][2] * vector.z;
	pResult->z = matrix.m[2][0] * vector.x + matrix.m[2][1] * vector.y + matrix.m[2][2] * vector.z;
}

// === Vector3 ===

void TRM_Vector3_negate(struct TRM_Vector3* pResult)
{
	pResult->x = -pResult->x;
	pResult->y = -pResult->y;
	pResult->z = -pResult->z;
}

void TRM_Vector3_addVector3(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, struct TRM_Vector3* pResult)
{
	pResult->x = vector0.x + vector1.x;
	pResult->y = vector0.y + vector1.y;
	pResult->z = vector0.z + vector1.z;
}

void TRM_Vector3_subVector3(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, struct TRM_Vector3* pResult)
{
	pResult->x = vector0.x - vector1.x;
	pResult->y = vector0.y - vector1.y;
	pResult->z = vector0.z - vector1.z;
}

void TRM_Vector3_addScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult)
{
	pResult->x = vector.x + scalar;
	pResult->y = vector.y + scalar;
	pResult->z = vector.z + scalar;
}

void TRM_Vector3_subScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult)
{
	pResult->x = vector.x - scalar;
	pResult->y = vector.y - scalar;
	pResult->z = vector.z - scalar;
}

void TRM_Vector3_multiplyWithScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult)
{
	pResult->x = vector.x * scalar;
	pResult->y = vector.y * scalar;
	pResult->z = vector.z * scalar;
}

void TRM_Vector3_divByScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult)
{
	pResult->x = vector.x / scalar;
	pResult->y = vector.y / scalar;
	pResult->z = vector.z / scalar;
}

void TRM_Vector3_getNorm2(struct TRM_Vector3 vector, float* pResult)
{
	*pResult = (vector.x * vector.x) + (vector.y * vector.y) + (vector.z * vector.z);
}

void TRM_Vector3_getNorm(struct TRM_Vector3 vector, float* pResult)
{
	float norm2;
	TRM_Vector3_getNorm2(vector, &norm2);
	*pResult = sqrtf(norm2);
}

void TRM_Vector3_normalize(struct TRM_Vector3* pResult)
{
	float norm;
	TRM_Vector3_getNorm(*pResult, &norm);

	if(norm == 0.0f)
	{
		exit(EXIT_FAILURE);
	}

	pResult->x = pResult->x / norm;
	pResult->y = pResult->y / norm;
	pResult->z = pResult->z / norm;
}

void TRM_Vector3_dot(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, float* pResult)
{
	*pResult = (vector0.x * vector1.x) + (vector0.y * vector1.y) + (vector0.z * vector1.z);
}

void TRM_Vector3_cross(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, struct TRM_Vector3* pResult)
{
	pResult->x = vector0.y * vector1.z - vector0.z * vector1.y;
	pResult->y = vector0.z * vector1.x - vector0.x * vector1.z;
	pResult->z = vector0.x * vector1.y - vector0.y * vector1.x;
}

// === utils ===

void TRM_getDirectionFromAngles(float yaw, float pitch, float roll, struct TRM_Vector3* pResult)
{
	struct TRM_Vector3 direction = {
		0.0f,
		0.0f,
		1.0f
	};

	struct TRM_Matrix4x4 rotation;
	TRM_Matrix4x4_getRotation(yaw, pitch, roll, &rotation);

	struct TRM_Matrix3x3 rotationUpper;
	TRM_Matrix4x4_getUpperLeft(rotation, &rotationUpper);
	TRM_Matrix3x3_multiplyWithVector3(rotationUpper, direction, pResult);
	TRM_Vector3_normalize(pResult);
}