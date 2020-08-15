#ifndef CODEGEN_HPP
#define CODEGEN_HPP

#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Pass.h>
#include <llvm/Support/FormattedStream.h>

#include <cmath>
#include <core/tree.hpp>

namespace Operon {
class CodeGen {
public:
    CodeGen()  {
        module = std::make_unique<llvm::Module>("codegen", context);
    }

    llvm::Function* createFunction(const std::string& name, int arity, llvm::Module* module) {
        auto func = module->getFunction(name);

        if (!func) {
            auto returnType = llvm::Type::getDoubleTy(module->getContext());
            std::vector<llvm::Type*> argTypes(arity, llvm::Type::getDoubleTy(module->getContext()));
            auto functionType = llvm::FunctionType::get(returnType, argTypes, false);
            func = llvm::Function::Create(functionType, llvm::Function::ExternalLinkage, name, module);
        }
        return func;
    }

    llvm::Function* CompileTree(const Operon::Tree& tree) {
        // we need to insert functions in the module for the mathematical operations that are not intrinsic
        auto tangent = createFunction("tan", 1, module.get()); 

        // we have to set up our function:
        // return type: a double value
        // input args:
        // - a double pointer representing our data
        // - number of rows
        // - number of columns
        // - data row we are evaluating right now
        std::vector<llvm::Type*> arguments;

        arguments.push_back(llvm::Type::getDoublePtrTy(module->getContext())); // data (pointer to double)
        arguments.push_back(llvm::Type::getInt32Ty(module->getContext())); // number of rows (int32)
        arguments.push_back(llvm::Type::getInt32Ty(module->getContext())); // number of cols (int32)
        arguments.push_back(llvm::Type::getInt32Ty(module->getContext())); // current row    (int32)

        auto fType = llvm::FunctionType::get(llvm::Type::getDoubleTy(module->getContext()), arguments, /* is var arg */ false);

        auto expr = llvm::Function::Create(fType, llvm::GlobalValue::ExternalLinkage, "expression", module.get());
        expr->setCallingConv(llvm::CallingConv::C);

        auto bb = llvm::BasicBlock::Create(module->getContext(), "entry", expr);
        llvm::IRBuilder<> builder(module->getContext());
        builder.SetInsertPoint(bb);

        // tree evaluation goes here
        const auto& nodes = tree.Nodes();
        std::vector<llvm::Value*> ops(nodes.size());

        for (size_t i = 0; i < nodes.size(); ++i) {
            switch (auto const& s = nodes[i]; s.Type) {
            case NodeType::Add: {
                auto a = i - 1; // first child index
                auto b = a - 1 - nodes[a].Length;
                ops[i] = builder.CreateBinOp(llvm::Instruction::Add, ops[a], ops[b]);
                break;
            }
            case NodeType::Sub: {
                auto a = i - 1; // first child index
                auto b = a - 1 - nodes[a].Length;
                ops[i] = builder.CreateBinOp(llvm::Instruction::Sub, ops[a], ops[b]);
                break;
            }
            case NodeType::Mul: {
                auto a = i - 1; // first child index
                auto b = a - 1 - nodes[a].Length;
                ops[i] = builder.CreateBinOp(llvm::Instruction::Mul, ops[a], ops[b]);
                break;
            }
            case NodeType::Div: {
                auto a = i - 1; // first child index
                auto b = a - 1 - nodes[a].Length;
                ops[i] = builder.CreateBinOp(llvm::Instruction::FDiv, ops[a], ops[b]);
                break;
            }
            case NodeType::Exp: {
                auto a = i - 1; // first child index
                ops[i] = builder.CreateUnaryIntrinsic(llvm::Intrinsic::exp, ops[a]);
                break;
            }
            case NodeType::Log: {
                auto a = i - 1; // first child index
                ops[i] = builder.CreateUnaryIntrinsic(llvm::Intrinsic::log, ops[a]);
                break;
            }
            case NodeType::Sin: {
                auto a = i - 1; // first child index
                ops[i] = builder.CreateUnaryIntrinsic(llvm::Intrinsic::sin, ops[a]);
                break;
            }
            case NodeType::Cos: {
                auto a = i - 1; // first child index
                ops[i] = builder.CreateUnaryIntrinsic(llvm::Intrinsic::cos, ops[a]);
                break;
            }
            case NodeType::Tan: {
                auto a = i - 1; // first child index
                llvm::SmallVector<llvm::Value*, 1> args { ops[a] };
                ops[i] = builder.CreateCall(tangent, args);
                break;
            }
            case NodeType::Variable: {
                break;
            }
            case NodeType::Constant: {
                ops[i] = llvm::ConstantFP::get(module->getContext(), llvm::APFloat(nodes[i].Value));
                break;
            }
            default: {
                break;
            }
            }
        }
    }

    void Evaluate(const Tree& tree)
    {
    }

private:
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module;
};
}

#endif
