#ifndef TRM_MATHS
#define TRM_MATHS

struct TRM_Matrix4x4
{
	float m[4][4];
};

struct TRM_Matrix3x3
{
	float m[3][3];
};

struct TRM_Vector3
{
	float x;
	float y;
	float z;
};

struct TRM_Vector4
{
	float x;
	float y;
	float z;
	float w;
};

// === Matrix4x4 ===

void TRM_Matrix4x4_transpose(struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_getUpperLeft(struct TRM_Matrix4x4 matrix, struct TRM_Matrix3x3* pResult);

void TRM_Matrix4x4_multiplyWithMatrix4x4(struct TRM_Matrix4x4 matrix0, struct TRM_Matrix4x4 matrix1, struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_multiplyWithVector4(struct TRM_Matrix4x4 matrix, struct TRM_Vector4 vector, struct TRM_Vector4* pResult);

void TRM_Matrix4x4_getIdentity(struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_getScaling(float scalingX, float scalingY, float scalingZ, struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_getRotation(float yaw, float pitch, float roll, struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_getInverseRotation(float yaw, float pitch, float roll, struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_getTranslation(struct TRM_Vector3 position, struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_getView(struct TRM_Vector3 position, float yaw, float pitch, float roll, struct TRM_Matrix4x4* pResult);

void TRM_Matrix4x4_getProjection(float fov, float aspectRatio, struct TRM_Matrix4x4* pResult);

// === Matrix3x3 ===

void TRM_Matrix3x3_multiplyWithMatrix3x3(struct TRM_Matrix3x3 matrix0, struct TRM_Matrix3x3 matrix1, struct TRM_Matrix3x3* pResult);

void TRM_Matrix3x3_multiplyWithVector3(struct TRM_Matrix3x3 matrix, struct TRM_Vector3 vector, struct TRM_Vector3* pResult);

// === vector3 ===

void TRM_Vector3_negate(struct TRM_Vector3* pResult);

void TRM_Vector3_addVector3(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, struct TRM_Vector3* pResult);

void TRM_Vector3_subVector3(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, struct TRM_Vector3* pResult);

void TRM_Vector3_addScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult);

void TRM_Vector3_subScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult);

void TRM_Vector3_multiplyWithScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult);

void TRM_Vector3_divByScalar(struct TRM_Vector3 vector, float scalar, struct TRM_Vector3* pResult);

void TRM_Vector3_getNorm2(struct TRM_Vector3 vector, float* pResult);

void TRM_Vector3_getNorm(struct TRM_Vector3 vector, float* pResult);

void TRM_Vector3_normalize(struct TRM_Vector3* pResult);

void TRM_Vector3_dot(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, float* pResult);

void TRM_Vector3_cross(struct TRM_Vector3 vector0, struct TRM_Vector3 vector1, struct TRM_Vector3* pResult);

// === utils ===

void TRM_getDirectionFromAngles(float yaw, float pitch, float roll, struct TRM_Vector3* pResult);

#endif