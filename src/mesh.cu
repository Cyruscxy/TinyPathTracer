#include "mesh.cuh"
#include <fstream>
#include <sstream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>
#include <tinygltf/json.hpp>

namespace 
{
    class VertexIndex {
    public:
        VertexIndex() = default;
        VertexIndex(long long int v, long long int vt, long long int vn) : position(v), uv(vt), normal(vn) {}

        bool operator<(const VertexIndex& i) const {
            if (position < i.position) return true;
            if (position > i.position) return false;
            if (uv < i.uv) return true;
            if (uv > i.uv) return false;
            if (normal < i.normal) return true;
            if (normal > i.normal) return false;
            return false;
        }

        long long int position = -1;
        long long int uv = -1;
        long long int normal = -1;
    };

    VertexIndex parseFaceVertexIndex(const std::string& token) {
        std::stringstream in(token);
        std::string VertexIndexString;
        int indices[3] = { 1, 1, 1 };

        int i = 0;
        while (std::getline(in, VertexIndexString, '/')) {
            if (VertexIndexString != "\\") {
                std::stringstream ss(VertexIndexString);
                ss >> indices[i];
                i++;
            }
        }

        return VertexIndex(indices[0] - 1, indices[1] - 1, indices[2] - 1);
    }

    inline std::istream& operator>>(std::istream& is, Vec3& v) { is >> v.x >> v.y >> v.z; return is; }

    struct addOffset
    {
        addOffset(uint32_t off) : offset(off) {}

        uint32_t offset;

        __host__ __device__
            uint32_t operator()(uint32_t indices)
        {
            indices += offset;
            return indices;
        }
    };
}


Scene::Scene(const std::string& filename, const std::string& type)
{
    if ( type == "gltf" )
    {
        readFromGLTF(filename);
    }
    else
    {
        throw std::runtime_error("Unsupported file format. Only support .gltf file.");
    }
}

void Scene::readFromGLTF(const std::string& filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    loader.SetStoreOriginalJSONForExtrasAndExtensions(true);
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

    if (!ret)
    {
        std::cerr << warn << std::endl;
        std::cerr << err << std::endl;
        throw std::runtime_error("Failed to read gltf file.");
    }

    std::vector<std::string> extensions;
    for ( auto ext : model.extensionsUsed )
    {
        extensions.push_back(ext);
    }

    for (auto node : model.nodes)
    {
        if (node.camera > -1)
        {
            auto& cameraInfo = model.cameras[node.camera];
            if ( cameraInfo.type == "perspective")
            {
                Real yFov = static_cast<Real>(cameraInfo.perspective.yfov); // in rad
                Real aspectRatio = static_cast<Real>(cameraInfo.perspective.aspectRatio);
                Real nearPlane = static_cast<Real>(cameraInfo.perspective.znear);

                auto& rQuat = node.rotation;
                auto& s = node.scale;
                auto& loc = node.translation;
                Quat rotationQuat;
                Vec3 scale;
                Vec3 translation;
                if ( !rQuat.empty() )
                {
                    rotationQuat = Quat(
                        static_cast<Real>(rQuat[0]),
                        static_cast<Real>(rQuat[1]),
                        static_cast<Real>(rQuat[2]),
                        static_cast<Real>(rQuat[3]));
                }
                if ( !s.empty() )
                {
                    scale = Vec3(
                        static_cast<Real>(s[0]),
                        static_cast<Real>(s[1]),
                        static_cast<Real>(s[2]));
                }
                else
                {
                    scale = Vec3(1.0f);
                }
                if ( !loc.empty() )
                {
                    translation = Vec3(
						static_cast<Real>(loc[0]),
						static_cast<Real>(loc[1]),
						static_cast<Real>(loc[2]));
                }

                m_camera = Camera(yFov, aspectRatio, nearPlane, translation, rotationQuat, scale);
            }
            else if ( cameraInfo.type == "orthographic")
            {
	            // TODO: Add orthographic camera implementation
            }
        }
        else if ( node.mesh > -1 )
        {
			tinygltf::Mesh& mesh = model.meshes[node.mesh];
        	m_meshes.push_back({});
            auto& currentObj = m_meshes.back();

            // read vertices coordinates into mesh
			tinygltf::Accessor& accessor = model.accessors[mesh.primitives[0].attributes["POSITION"]];
			tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
			tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            size_t count = accessor.count;
			float* position = reinterpret_cast<float*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
            currentObj.readVerticesFromRawPtr(position, count);

            // read indices into mesh
            accessor = model.accessors[mesh.primitives[0].indices];
            bufferView = model.bufferViews[accessor.bufferView];
            buffer = model.buffers[bufferView.buffer];
            count = accessor.count;
            if ( accessor.componentType == 5120 )
            {
                int8_t* indices = reinterpret_cast<int8_t*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
                currentObj.readIndicesFromRawPtr(indices, count);
            }
            else if ( accessor.componentType == 5121 )
            {
                uint8_t* indices = reinterpret_cast<uint8_t*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
                currentObj.readIndicesFromRawPtr(indices, count);
            }
            else if ( accessor.componentType == 5122 )
            {
                int16_t* indices = reinterpret_cast<int16_t*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
                currentObj.readIndicesFromRawPtr(indices, count);
            }
            else if ( accessor.componentType == 5123 )
            {
                uint16_t* indices = reinterpret_cast<uint16_t*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
                currentObj.readIndicesFromRawPtr(indices, count);
            }
            else if ( accessor.componentType == 5124 )
            {
                int32_t* indices = reinterpret_cast<int32_t*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
                currentObj.readIndicesFromRawPtr(indices, count);
            }
            else if (accessor.componentType == 5125)
            {
                uint32_t* indices = reinterpret_cast<uint32_t*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
                currentObj.readIndicesFromRawPtr(indices, count);
            }

            // read normals into mesh
            accessor = model.accessors[mesh.primitives[0].attributes["NORMAL"]];
            bufferView = model.bufferViews[accessor.bufferView];
            buffer = model.buffers[bufferView.buffer];
            count = accessor.count;
            float* normals = reinterpret_cast<float*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
            currentObj.readNormalsFromRawPtr(normals, count);

            // read texture coords into mesh
            accessor = model.accessors[mesh.primitives[0].attributes["TEXCOORD_0"]];
            bufferView = model.bufferViews[accessor.bufferView];
            buffer = model.buffers[bufferView.buffer];
            count = accessor.count;
            float* texcoords = reinterpret_cast<float*>(&(buffer.data[accessor.byteOffset + bufferView.byteOffset]));
            currentObj.readTexcoordsFromRawPtr(texcoords, count);

            // read material
            if ( !model.materials.empty() )
            {
                auto material = model.materials[mesh.primitives[0].material];
                if (m_materials.find(material.name) == m_materials.end())
                {
                    auto& currentMaterial = m_materials[material.name];
                    currentMaterial.roughness = static_cast<Real>(material.pbrMetallicRoughness.roughnessFactor);
                    currentMaterial.metallic = static_cast<Real>(material.pbrMetallicRoughness.metallicFactor);
                    currentMaterial.baseColor = Vec3(
                        static_cast<Real>(material.pbrMetallicRoughness.baseColorFactor[0]),
                        static_cast<Real>(material.pbrMetallicRoughness.baseColorFactor[1]),
                        static_cast<Real>(material.pbrMetallicRoughness.baseColorFactor[2]));

                    if (!material.extensions_json_string.empty())
                    {
                        using json = nlohmann::json;
                        json extension = json::parse(material.extensions_json_string);

                        for (auto item : extension.items())
                        {
                            if (item.key() == "KHR_materials_transmission")
                            {
                                currentMaterial.specular = 1.0f - static_cast<Real>(item.value()["transmissionFactor"]) / 5.0f;
                            }
                            if (item.key() == "KHR_materials_emissive_strength")
                            {
                                currentMaterial.emissionFactor = item.value()["emissiveStrength"];
                            }
                            if (item.key() == "KHR_materials_ior")
                            {
                                currentMaterial.eta = item.value()["ior"];
                            }
                        }
                    }
                }
                currentObj.m_material = material.name;
            }

            // read transform 
            auto& rQuat = node.rotation;
            auto& s = node.scale;
            auto& loc = node.translation;
            Quat rotationQuat;
            Vec3 scale;
            Vec3 translation;
            if (!rQuat.empty())
            {
                rotationQuat = Quat(
                    static_cast<Real>(rQuat[0]),
                    static_cast<Real>(rQuat[1]),
                    static_cast<Real>(rQuat[2]),
                    static_cast<Real>(rQuat[3]));
            }
            if (!s.empty())
            {
                scale = Vec3(
                    static_cast<Real>(s[0]),
                    static_cast<Real>(s[1]),
                    static_cast<Real>(s[2]));
            }
            else
            {
                scale = Vec3(1.0f);
            }
            if (!loc.empty())
            {
                translation = Vec3(
                    static_cast<Real>(loc[0]),
                    static_cast<Real>(loc[1]),
                    static_cast<Real>(loc[2]));
            }
            currentObj.m_transform = std::make_shared<Transform>(translation, rotationQuat, scale);
            // TODO: Hierarchy transform support
        }
    }
}

DeviceScene Scene::copySceneToDevice()
{
    // calculate total count of elements
    size_t indicesCount = 0;
    size_t verticesCount = 0;
    for ( auto& mesh : m_meshes )
    {
        indicesCount += mesh.m_polygons.size();
        verticesCount += mesh.m_vertexCoords.size();
    }

    // allocate device memory for scene elements
    DeviceScene deviceScene(verticesCount, indicesCount, m_meshes.size(), m_materials.size());

    uint32_t currentIndicesCount = 0;
    uint32_t currentVerticesCount = 0;

    std::map<std::string, uint32_t> materialsIndices;
    uint32_t i = 0;
    for ( auto& [name, mtl] : m_materials)
    {
        deviceScene.materials[i] = mtl;
        materialsIndices[name] = i;
        i += 1;
    }

    i = 0;
    std::vector<MtlInterval> hostMtlLUT(m_meshes.size());
    for ( auto& mesh : m_meshes )
    {
        // copy indices
        thrust::copy(mesh.m_polygons.begin(), mesh.m_polygons.end(), 
            deviceScene.indices.begin() + currentIndicesCount);
        thrust::transform(
            deviceScene.indices.begin() + currentIndicesCount, 
            deviceScene.indices.begin() + currentIndicesCount + static_cast<uint32_t>(mesh.m_polygons.size()),
            deviceScene.indices.begin() + currentIndicesCount,
            addOffset(currentVerticesCount));

        // copy vertex attributes
        thrust::copy(mesh.m_vertexCoords.begin(), mesh.m_vertexCoords.end(),
            deviceScene.vertices.begin() + currentVerticesCount);
        thrust::copy(mesh.m_vertexNormals.begin(), mesh.m_vertexNormals.end(),
            deviceScene.normals.begin() + currentVerticesCount);
        thrust::copy(mesh.m_vertexTexCoords.begin(), mesh.m_vertexTexCoords.end(),
            deviceScene.texCoords.begin() + currentVerticesCount);

        // record the material used by current mesh
        hostMtlLUT[i].begin = currentIndicesCount / 3;
        hostMtlLUT[i].mtlIdx = materialsIndices[mesh.m_material];

        currentIndicesCount += static_cast<uint32_t>(mesh.m_polygons.size());
        currentVerticesCount += static_cast<uint32_t>(mesh.m_vertexCoords.size());
        i += 1;
    }

    thrust::copy(hostMtlLUT.begin(), hostMtlLUT.end(), deviceScene.materialsLUT.begin());

    // copy transform matrices
    std::vector<Mat4> vertTrans;
    std::vector<Mat4> normalTrans;

    auto normal_to_world = [](Mat4 const& l2w)-> Mat4 {
        auto m = Mat4{
                Vec4 { l2w[0][0], l2w[0][1], l2w[0][2], 0.0f },
                Vec4 { l2w[1][0], l2w[1][1], l2w[1][2], 0.0f },
                Vec4 { l2w[2][0], l2w[2][1], l2w[2][2], 0.0f },
                Vec4 {      0.0f,      0.0f,      0.0f, 1.0f } };
        return m.transpose().inverse();
    };
    for ( auto& m : m_meshes )
    {
        vertTrans.push_back(m.m_transform->localToWorld());
        auto n2w = normal_to_world(vertTrans.back());
        normalTrans.push_back(n2w);
    }
    thrust::copy(vertTrans.begin(), vertTrans.end(), deviceScene.vertTrans.begin());
    thrust::copy(normalTrans.begin(), normalTrans.end(), deviceScene.normalTrans.begin());

    return deviceScene;
}
